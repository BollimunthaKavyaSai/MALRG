require("dotenv").config();

const express = require("express");
const fs = require("fs");
const path = require("path");
const cors = require("cors");
const mongoose = require("mongoose");

const app = express();

app.use(cors());
app.use(express.json({ limit: "10mb" }));

const PORT = process.env.PORT || 3000;
// must match the AI engine's limit (GenerationRequest.max_papers in app.py)
const MAX_PAPERS = 20;

const MONGODB_URI =
  process.env.MONGODB_URI || "mongodb://127.0.0.1:27017/malrg";

const PYTHON_AI_URL =
  process.env.PYTHON_AI_URL || "http://127.0.0.1:8000";

/*
 * |--------------------------------------------------------------------------
 * | MongoDB Schemas
 * |--------------------------------------------------------------------------
 */

const paperSchema = new mongoose.Schema(
  {
    paper_id: { type: String, required: true, unique: true },
    title: String,
    authors: [String],
    year: Number,
    publication_date: String,
    abstract: String,
    doi: String,
    venue: String,
    publication_types: [String],
    url: String,
    pdf_url: String,
    source: String,
    is_open_access: Boolean,
    is_preprint: Boolean,
    is_available: Boolean,

    methodology: String,
    results: String,
    evaluation_metrics: [String],
    conclusion: String,
    limitations: String,

    retrieval_queries: [String]
  },
  {
    timestamps: true
  }
);

const runSchema = new mongoose.Schema(
  {
    topic: { type: String, required: true },
    start_year: Number,
    end_year: Number,
    max_papers: Number,
    python_task_id: String,

    status: {
      type: String,
      default: "created"
    },

    research_gap: mongoose.Schema.Types.Mixed,

    final_review: String,
    final_review_md: String,
    raw_papers_list: [mongoose.Schema.Types.Mixed],
    papers: [mongoose.Schema.Types.Mixed],

    evaluation: mongoose.Schema.Types.Mixed,

    // per-agent progress events reported by the AI engine
    progress: [mongoose.Schema.Types.Mixed],

    // human feedback rounds: comments + critic scores before / after revision
    feedback_history: [mongoose.Schema.Types.Mixed],

    data: mongoose.Schema.Types.Mixed,

    error: String,
    reason: String,
    retryable: Boolean
  },
  {
    timestamps: true,
    strict: false
  }
);

paperSchema.index({
  title: "text",
  abstract: "text",
  venue: "text"
});

const Paper = mongoose.model("Paper", paperSchema);
const Run = mongoose.model("Run", runSchema);

/*
 * |--------------------------------------------------------------------------
 * | Utility Functions
 * |--------------------------------------------------------------------------
 */

/**
 * Normalize incoming academic search queries.
 *
 * Supported:
 *
 * 1. JSON array:
 *    ["query one", "query two"]
 *
 * 2. Pipe separated:
 *    query one|query two
 *
 * 3. Repeated query parameters:
 *    ?queries=query1&queries=query2
 *
 * 4. Single string:
 *    query one
 *
 * The Python Query Understanding agent is expected to
 * provide the expanded academic queries.
 */
// Cleans the 'queries' value that Python sends to /api/corpus/search into a plain list of search phrases.
// The input may be an array, a JSON text like '["a","b"]', or a text separated by '|'.
// Steps: turn it into a list; if it is empty use the topic itself; trim spaces; drop empty or duplicate phrases
// (ignoring upper/lower case); keep at most 20 phrases.
//Python sends search phrases in different shapes. This turns them into one clean list, with no empty or repeated phrases.
function normalizeSearchQueries(rawQueries, topic) {
  let queries = [];

  if (Array.isArray(rawQueries)) {
    queries = rawQueries;
  } else if (typeof rawQueries === "string") {
    const value = rawQueries.trim();

    if (value) {
      try {
        const parsed = JSON.parse(value);

        if (Array.isArray(parsed)) {
          queries = parsed;
        } else {
          queries = value.includes("|")
            ? value.split("|")
            : [value];
        }
      } catch (error) {
        queries = value.includes("|")
          ? value.split("|")
          : [value];
      }
    }
  }

  if (!queries.length && topic) {
    queries = [topic];
  }

  const cleaned = [];
  const seen = new Set();

  for (const query of queries) {
    if (typeof query !== "string") {
      continue;
    }

    const normalized = query
      .replace(/\s+/g, " ")
      .trim();

    if (!normalized) {
      continue;
    }

    const key = normalized.toLowerCase();

    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    cleaned.push(normalized);
  }

  return cleaned.slice(0, 20);
}

/**
 * Stable paper deduplication key.
 */
// Makes a unique key for a paper so the same paper found by several queries is counted only once.
// Priority: 1) paper_id, 2) DOI (cleaned), 3) title + year (lowercase, punctuation removed).
// Returns null if the paper has none of these.
//	Makes a “name tag” for a paper (id, or DOI, or title+year), so the same paper is not counted twice.
function paperDedupKey(paper) {
  if (!paper) {
    return null;
  }

  const paperId = String(paper.paper_id || "").trim();

  if (paperId) {
    return `id:${paperId.toLowerCase()}`;
  }

  const doi = String(paper.doi || "")
    .trim()
    .toLowerCase()
    .replace(/^https?:\/\/doi\.org\//, "")
    .replace(/^doi:/, "");

  if (doi) {
    return `doi:${doi}`;
  }

  const title = String(paper.title || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const year = paper.year || "";

  if (title) {
    return `title:${title}:${year}`;
  }

  return null;
}

/**
 * Escape a string before creating a MongoDB regex.
 */
// Puts a backslash before special regex characters (. * + ? ^ $ { } ( ) | [ ] \) so that normal text
// typed by the user can be used safely inside a MongoDB regular-expression search.
//Makes special characters safe, so they don’t break a search.
function escapeRegex(value) {
  return String(value || "").replace(
    /[.*+?^${}()|[\]\\]/g,
    "\\$&"
  );
}

/**
 * Extract meaningful terms from an academic query.
 *
 * This is only a secondary candidate-discovery mechanism.
 * It is NOT the final relevance ranking.
 */
// Picks the meaningful single words from a search query: lowercases it, removes filler words (a, and, for, of ...),
// removes very short words and duplicates, and keeps at most 15 words.
// These words are used for the second (word-matching) search against title, abstract and venue.
//Takes the important single words from a query and drops filler words like “the” and “of”.
function extractSearchTerms(query) {
  const stopWords = new Set([
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "with",
    "based",
    "approach",
    "approaches",
    "study",
    "studies",
    "review",
    "reviews",
    "system",
    "systems",
    "method",
    "methods",
    "framework",
    "frameworks"
  ]);

  const words = String(query || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .map((word) => word.trim())
    .filter(Boolean)
    .filter((word) => word.length >= 3)
    .filter((word) => !stopWords.has(word));

  return [...new Set(words)].slice(0, 15);
}

/**
 * Extract stronger concept phrases from a query.
 *
 * Examples:
 *
 * "multi-agent literature review generation"
 *
 * becomes:
 *
 * "multi-agent"
 * "literature review"
 * "review generation"
 * "literature review generation"
 */
// Finds important phrases in a query, for example 'multi-agent', 'literature review' or 'review generation'.
// Steps: 1) normalise the text; 2) add every phrase from a built-in list of important phrases that appears in the query;
// 3) add every pair of neighbouring words that are both at least 4 letters long; 4) keep at most 20 phrases.
// Used by calculateCandidateConceptScore to order candidates.
//Finds important phrases like “multi-agent” or “literature review”.
function extractConceptPhrases(query) {
  const normalized = String(query || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (!normalized) {
    return [];
  }

  const phrases = new Set();

  const importantPhrases = [
    "multi agent",
    "multi-agent",
    "multiagent",
    "literature review",
    "review generation",
    "literature review generation",
    "automated literature review",
    "automatic literature review",
    "llm literature review",
    "language model literature review",
    "research assistant",
    "research assistants",
    "scientific literature",
    "scientific literature synthesis",
    "citation grounded",
    "evidence grounded",
    "citation aware",
    "scholarly information retrieval",
    "literature retrieval",
    "literature synthesis",
    "scientific question answering",
    "ai agents",
    "llm agents",
    "multi-agent systems",
    "multi-agent system"
  ];

  for (const phrase of importantPhrases) {
    if (normalized.includes(phrase)) {
      phrases.add(phrase);
    }
  }

  const words = normalized.split(/\s+/);

  for (let i = 0; i < words.length - 1; i++) {
    const phrase = `${words[i]} ${words[i + 1]}`;

    if (
      words[i].length >= 4 &&
      words[i + 1].length >= 4
    ) {
      phrases.add(phrase);
    }
  }

  return [...phrases].slice(0, 20);
}

/**
 * Calculate a small deterministic lexical relevance score.
 *
 * This score is only used to improve candidate ordering.
 * Python performs the actual semantic + lexical ranking.
 */
// Gives a paper a small score for how well it matches the search queries (only used to ORDER candidates;
// Python does the real ranking later).
// For each query: every concept phrase found in the paper scores 5 if it is in the title, otherwise 2;
// every search word scores 2 if it is in the title, otherwise 1 if it is in the abstract.
//Gives each paper a small score for how well it matches the query words. Used only to order papers.
function calculateCandidateConceptScore(paper, searchQueries) {
  const title = String(paper.title || "").toLowerCase();
  const abstract = String(paper.abstract || "").toLowerCase();
  const venue = String(paper.venue || "").toLowerCase();

  const searchable = `${title} ${abstract} ${venue}`;

  let score = 0;

  for (const query of searchQueries) {
    const normalizedQuery = String(query || "")
      .toLowerCase()
      .replace(/\s+/g, " ")
      .trim();

    if (!normalizedQuery) {
      continue;
    }

    const phrases = extractConceptPhrases(normalizedQuery);

    for (const phrase of phrases) {
      if (searchable.includes(phrase)) {
        score += title.includes(phrase) ? 5 : 2;
      }
    }

    const terms = extractSearchTerms(normalizedQuery);

    for (const term of terms) {
      if (title.includes(term)) {
        score += 2;
      } else if (abstract.includes(term)) {
        score += 1;
      }
    }
  }

  return score;
}

/*
 * |--------------------------------------------------------------------------
 * | Health Check
 * |--------------------------------------------------------------------------
 */

// GET /api/health - tells the browser and the start scripts whether the Node server is working.
// Returns ok=true, whether MongoDB is connected (mongoose connection state 1), and the Python AI engine URL.
//“Are you alive?” It answers yes, and whether MongoDB is connected.
app.get("/api/health", async (req, res) => {
  let mongodb = false;

  try { 
    mongodb = mongoose.connection.readyState === 1;
  } catch (error) {
    mongodb = false;
  }

  res.json({
    ok: true,
    backend: "node-express",
    mongodb,
    python_ai_url: PYTHON_AI_URL
  });
});

/*
 * |--------------------------------------------------------------------------
 * | Corpus Statistics
 * |--------------------------------------------------------------------------
 */

// GET /api/corpus/stats - numbers shown on the 'Paper Corpus' tab and its badge.
// Counts all papers, counts the eligible ones (not a preprint and available), and finds the oldest and newest year.
//“How many papers do you have?” It counts the papers and finds the oldest and newest year.
app.get("/api/corpus/stats", async (req, res) => {
  try {
    const total = await Paper.countDocuments();

    const published = await Paper.countDocuments({
      is_preprint: false,
      is_available: true
    });

    const years = await Paper.aggregate([
      {
        $match: {
          year: { $exists: true }
        }
      },
      {
        $group: {
          _id: null,
          min_year: { $min: "$year" },
          max_year: { $max: "$year" }
        }
      }
    ]);

    res.json({
      total,
      eligible_published: published,
      year_range: years.length
        ? {
            min: years[0].min_year,
            max: years[0].max_year
          }
        : null
    });
  } catch (error) {
    console.error("Corpus stats error:", error);

    res.status(500).json({
      error: "Failed to retrieve corpus statistics"
    });
  }
});

/*
 * |--------------------------------------------------------------------------
 * | Paginated Corpus Papers for Database Explorer
 * |--------------------------------------------------------------------------
 */
// GET /api/corpus/papers - the paginated list used by the Paper Corpus explorer.
// Reads page, limit (max 100), optional start_year/end_year and a search text q.
// Builds a MongoDB filter (eligible papers only; q is matched against title, venue, abstract and methodology),
// counts the matches, sorts by newest year, skips/limits for the page and returns papers + total + totalPages.
// The list for the Paper Corpus tab, page by page, with search.
app.get("/api/corpus/papers", async (req, res) => {
  try {
    const page = Math.max(1, parseInt(req.query.page) || 1);
    const limit = Math.min(100, Math.max(1, parseInt(req.query.limit) || 20));
    const skip = (page - 1) * limit;

    const filter = {
      is_available: true,
      is_preprint: false
    };

    if (req.query.start_year) {
      filter.year = filter.year || {};
      filter.year.$gte = parseInt(req.query.start_year);
    }
    if (req.query.end_year) {
      filter.year = filter.year || {};
      filter.year.$lte = parseInt(req.query.end_year);
    }

    if (req.query.q) {
      const q = req.query.q.trim();
      const regex = new RegExp(escapeRegex(q), "i");
      filter.$or = [
        { title: regex },
        { venue: regex },
        { abstract: regex },
        { methodology: regex }
      ];
    }

    const total = await Paper.countDocuments(filter);
    const papers = await Paper.find(filter)
      .sort({ year: -1, createdAt: -1 })
      .skip(skip)
      .limit(limit)
      .lean();

    res.json({
      success: true,
      total,
      page,
      limit,
      totalPages: Math.ceil(total / limit),
      papers
    });
  } catch (error) {
    console.error("Corpus papers listing error:", error);
    res.status(500).json({ error: "Failed to retrieve corpus papers" });
  }
});

/*
 * |--------------------------------------------------------------------------
 * | Benchmark & Ablation Study Metrics (Q1 Standard)
 * |--------------------------------------------------------------------------
 */
const ABLATION_RESULTS_DIR = path.join(
  __dirname,
  "..",
  "MALRG_Evaluation_Suite",
  "results"
);

// Real, computed ablation results (written by `python run_all.py analyze`
// in MALRG_Evaluation_Suite). Nothing here is hand-entered.
app.use(
  "/api/evaluation/charts",
  express.static(path.join(ABLATION_RESULTS_DIR, "figures"))
);

// GET /api/evaluation/benchmark - gives the Evaluation tab its data.
// Reads the computed ablation summary file benchmark.json from MALRG_Evaluation_Suite/results and lists the
// PNG chart files in its figures folder. If the file does not exist it answers available=false with a hint
// how to create it. (The charts themselves are served by the express.static line above.)
// Reads the results file of the evaluation and sends it to the Evaluation tab.
app.get("/api/evaluation/benchmark", async (req, res) => {
  try {
    const summaryPath = path.join(ABLATION_RESULTS_DIR, "benchmark.json");

    if (!fs.existsSync(summaryPath)) {
      return res.json({
        success: true,
        available: false,
        message:
          "No ablation results yet. Run: cd MALRG_Evaluation_Suite && python run_all.py all"
      });
    }

    const summary = JSON.parse(fs.readFileSync(summaryPath, "utf-8"));
    const charts = fs.existsSync(path.join(ABLATION_RESULTS_DIR, "figures"))
      ? fs
          .readdirSync(path.join(ABLATION_RESULTS_DIR, "figures"))
          .filter((name) => name.endsWith(".png"))
      : [];

    res.json({
      success: true,
      available: true,
      charts,
      ...summary
    });
  } catch (error) {
    console.error("Benchmark evaluation error:", error);
    res.status(500).json({ error: "Failed to load benchmark evaluation data" });
  }
});

/*
 * |--------------------------------------------------------------------------
 * | Store Papers
 * |--------------------------------------------------------------------------
 */

// POST /api/corpus/papers - saves papers into MongoDB (used by the seed script and by the OpenAlex online fallback).
// It accepts an array or {papers: [...]}. For each paper with a paper_id it does an 'upsert':
// update the paper if it already exists, otherwise insert it. Returns how many were inserted and updated.
// The list for the Paper Corpus tab, page by page, with search.Saves papers into MongoDB. If a paper exists it updates it, otherwise it inserts it.
app.post("/api/corpus/papers", async (req, res) => {
  try {
    const papers = Array.isArray(req.body)
      ? req.body
      : req.body.papers;

    if (!Array.isArray(papers)) {
      return res.status(400).json({
        error: "Expected an array of papers"
      });
    }

    let inserted = 0;
    let updated = 0;

    for (const paper of papers) {
      if (!paper.paper_id) {
        continue;
      }

      const existing = await Paper.findOne({
        paper_id: paper.paper_id
      });

      await Paper.findOneAndUpdate(
        { paper_id: paper.paper_id },
        paper,
        {
          upsert: true,
          new: true,
          setDefaultsOnInsert: true
        }
      );

      if (existing) {
        updated++;
      } else {
        inserted++;
      }
    }

    res.json({
      success: true,
      inserted,
      updated,
      total_received: papers.length
    });
  } catch (error) {
    console.error("Paper storage error:", error);

    res.status(500).json({
      error: "Failed to store papers"
    });
  }
});

/*
 * |--------------------------------------------------------------------------
 * | Search Local Scholarly Corpus
 * |--------------------------------------------------------------------------
 *
 * Retrieval pipeline:
 *
 * Python Query Understanding
 *          |
 *          v
 * Multiple academic queries
 *          |
 *          v
 * MongoDB multi-query candidate retrieval
 *          |
 *          v
 * Lexical fallback
 *          |
 *          v
 * Deduplication
 *          |
 *          v
 * Candidate ordering
 *          |
 *          v
 * Python hybrid semantic + lexical ranking
 *
 * IMPORTANT:
 *
 * This endpoint does NOT decide the final papers.
 * It deliberately returns a larger candidate pool.
 */

// GET /api/corpus/search - the endpoint the Python Search agent calls to get CANDIDATE papers from MongoDB.
// It does NOT pick the final papers; Python ranks them. Steps:
//  1. Read topic, queries, years and limit; clean the queries (normalizeSearchQueries).
//  2. Base filter: not a preprint, available, year between start and end.
//  3. For EVERY query run a MongoDB text search, keep the best text score, remember which queries matched,
//     and merge duplicates with paperDedupKey.
//  4. Second search: regex match of the important query words against title, abstract and venue,
//     to catch papers that use slightly different wording.
//  5. Score each candidate with calculateCandidateConceptScore.
//  6. Order by: number of queries matched, concept score, text score, newest year.
//  7. Return up to about 100-200 candidates (a big pool, so Python can reject adjacent-topic papers).
// Python asks: “Give me candidate papers for these queries.” Node searches MongoDB and returns a big list.
app.get("/api/corpus/search", async (req, res) => {
  try {
    const {
      topic = "",
      start_year,
      end_year,
      limit = 20
    } = req.query;

    const searchQueries = normalizeSearchQueries(
      req.query.queries,
      topic
    );

    const requestedLimit = Number(limit);

    const finalLimit =
      Number.isFinite(requestedLimit) && requestedLimit > 0
        ? Math.min(Math.floor(requestedLimit), 100)
        : 20;

    /*
     * ---------------------------------------------------------------
     * Validate year filters
     * ---------------------------------------------------------------
     */

    let minYear = null;
    let maxYear = null;

    if (start_year !== undefined && start_year !== "") {
      const parsedStart = Number(start_year);

      if (!Number.isInteger(parsedStart)) {
        return res.status(400).json({
          error: "start_year must be an integer"
        });
      }

      minYear = parsedStart;
    }

    if (end_year !== undefined && end_year !== "") {
      const parsedEnd = Number(end_year);

      if (!Number.isInteger(parsedEnd)) {
        return res.status(400).json({
          error: "end_year must be an integer"
        });
      }

      maxYear = parsedEnd;
    }

    if (
      minYear !== null &&
      maxYear !== null &&
      minYear > maxYear
    ) {
      return res.status(400).json({
        error: "start_year must be less than or equal to end_year"
      });
    }

    /*
     * ---------------------------------------------------------------
     * Base eligibility filter
     * ---------------------------------------------------------------
     *
     * Only available published records are eligible.
     *
     * Preprints are explicitly excluded.
     */

    const baseQuery = {
      is_preprint: false,
      is_available: true
    };

    if (minYear !== null) {
      baseQuery.year = {
        ...(baseQuery.year || {}),
        $gte: minYear
      };
    }

    if (maxYear !== null) {
      baseQuery.year = {
        ...(baseQuery.year || {}),
        $lte: maxYear
      };
    }

    /*
     * ---------------------------------------------------------------
     * No search query
     * ---------------------------------------------------------------
     */

    if (!searchQueries.length) {
      const papers = await Paper.find(baseQuery)
        .limit(finalLimit)
        .sort({
          year: -1,
          createdAt: -1
        })
        .lean();

      return res.json({
        count: papers.length,
        papers,
        search_queries: [],
        candidate_count: papers.length,
        retrieval_mode: "recent_eligible"
      });
    }

    /*
     * ---------------------------------------------------------------
     * Multi-query candidate retrieval
     * ---------------------------------------------------------------
     *
     * Retrieve significantly more candidates than requested.
     *
     * Example:
     *
     * finalLimit = 5
     *
     * candidateLimitPerQuery = 50
     *
     * If 10 expanded queries are supplied, each query can contribute
     * up to 50 candidates before deduplication.
     *
     * This is intentional because the Python ranker needs enough
     * candidates to distinguish directly relevant literature from
     * merely adjacent literature.
     */

    const candidateLimitPerQuery = Math.min(
      Math.max(finalLimit * 10, 50),
      100
    );

    const candidateMap = new Map();

    /*
     * ---------------------------------------------------------------
     * Search every expanded query independently
     * ---------------------------------------------------------------
     */

    for (const searchQuery of searchQueries) {
      try {
        const query = {
          ...baseQuery,
          $text: {
            $search: searchQuery
          }
        };

        const papers = await Paper.find(
          query,
          {
            score: {
              $meta: "textScore"
            }
          }
        )
          .sort({
            score: {
              $meta: "textScore"
            },
            year: -1
          })
          .limit(candidateLimitPerQuery)
          .lean();

        for (const paper of papers) {
          const key = paperDedupKey(paper);

          if (!key) {
            continue;
          }

          const existing = candidateMap.get(key);

          if (!existing) {
            candidateMap.set(key, {
              ...paper,
              _retrieval_score: Number(paper.score || 0),
              _matched_queries: [searchQuery]
            });

            continue;
          }

          const currentScore = Number(
            existing._retrieval_score || 0
          );

          const newScore = Number(
            paper.score || 0
          );

          if (newScore > currentScore) {
            existing._retrieval_score = newScore;
          }

          if (
            !existing._matched_queries.includes(searchQuery)
          ) {
            existing._matched_queries.push(searchQuery);
          }
        }
      } catch (queryError) {
        console.warn(
          `Mongo text query failed for "${searchQuery}":`,
          queryError.message
        );
      }
    }

    /*
     * ---------------------------------------------------------------
     * Secondary lexical fallback
     * ---------------------------------------------------------------
     *
     * MongoDB text search can miss useful candidates when terminology
     * differs slightly.
     *
     * Therefore we supplement text search with title/abstract/venue
     * matching for important terms.
     *
     * This is candidate discovery only.
     */

    const allTerms = [];

    for (const searchQuery of searchQueries) {
      allTerms.push(...extractSearchTerms(searchQuery));
    }

    const uniqueTerms = [
      ...new Set(allTerms)
    ].slice(0, 25);

    if (uniqueTerms.length > 0) {
      const regexAlternatives = uniqueTerms.map(
        (term) =>
          new RegExp(
            escapeRegex(term),
            "i"
          )
      );

      try {
        const fallbackQuery = {
          ...baseQuery,
          $or: [
            {
              title: {
                $in: regexAlternatives
              }
            },
            {
              abstract: {
                $in: regexAlternatives
              }
            },
            {
              venue: {
                $in: regexAlternatives
              }
            }
          ]
        };

        const fallbackPapers = await Paper.find(
          fallbackQuery
        )
          .sort({
            year: -1
          })
          .limit(
            Math.min(
              Math.max(finalLimit * 20, 100),
              300
            )
          )
          .lean();

        for (const paper of fallbackPapers) {
          const key = paperDedupKey(paper);

          if (!key) {
            continue;
          }

          const existing = candidateMap.get(key);

          if (!existing) {
            candidateMap.set(key, {
              ...paper,
              _retrieval_score: 0,
              _matched_queries: []
            });
          }
        }
      } catch (fallbackError) {
        console.warn(
          "Corpus lexical fallback failed:",
          fallbackError.message
        );
      }
    }

    /*
     * ---------------------------------------------------------------
     * Candidate ranking
     * ---------------------------------------------------------------
     *
     * Ranking priority:
     *
     * 1. Number of expanded queries matched
     * 2. Concept relevance
     * 3. MongoDB text relevance
     * 4. Publication year
     *
     * The purpose is to preserve papers supported by multiple
     * independent query formulations.
     *
     * Python remains responsible for the final hybrid ranking.
     */

    const candidates = Array.from(
      candidateMap.values()
    );

    for (const paper of candidates) {
      paper._concept_score =
        calculateCandidateConceptScore(
          paper,
          searchQueries
        );
    }

    candidates.sort((a, b) => {
      const coverageA = Array.isArray(
        a._matched_queries
      )
        ? a._matched_queries.length
        : 0;

      const coverageB = Array.isArray(
        b._matched_queries
      )
        ? b._matched_queries.length
        : 0;

      if (coverageA !== coverageB) {
        return coverageB - coverageA;
      }

      const conceptA = Number(
        a._concept_score || 0
      );

      const conceptB = Number(
        b._concept_score || 0
      );

      if (conceptA !== conceptB) {
        return conceptB - conceptA;
      }

      const scoreA = Number(
        a._retrieval_score || 0
      );

      const scoreB = Number(
        b._retrieval_score || 0
      );

      if (scoreA !== scoreB) {
        return scoreB - scoreA;
      }

      const yearA = Number(
        a.year || 0
      );

      const yearB = Number(
        b.year || 0
      );

      return yearB - yearA;
    });

    /*
     * ---------------------------------------------------------------
     * Return a LARGE candidate pool
     * ---------------------------------------------------------------
     *
     * Do not reduce this to finalLimit here.
     *
     * Python performs the final semantic + TF-IDF relevance ranking.
     *
     * Returning up to 100 candidates gives the Python ranker enough
     * material to reject adjacent topics such as:
     *
     * - smart vehicles
     * - smart cities
     * - healthcare agents
     * - generic agentic AI
     *
     * when the requested topic is specifically:
     *
     * "multi-agent literature review generation"
     */

    const maximumCandidatesToReturn = Math.min(
      Math.max(finalLimit * 20, 100),
      200
    );

    const selectedCandidates = candidates
      .slice(
        0,
        Math.min(
          candidates.length,
          maximumCandidatesToReturn
        )
      )
      .map((paper) => {
        const cleaned = {
          ...paper
        };

        delete cleaned.score;
        delete cleaned._retrieval_score;
        delete cleaned._matched_queries;
        delete cleaned._concept_score;

        return cleaned;
      });

    res.json({
      count: selectedCandidates.length,
      papers: selectedCandidates,
      search_queries: searchQueries,
      candidate_count: candidates.length,
      returned_candidate_count: selectedCandidates.length,
      requested_final_limit: finalLimit,
      retrieval_mode:
        "multi_query_text_plus_lexical_concept_candidate_retrieval"
    });
  } catch (error) {
    console.error(
      "Corpus search error:",
      error
    );

    res.status(500).json({
      error: "Failed to search scholarly corpus"
    });
  }
});

/*
 * |--------------------------------------------------------------------------
 * | Create Literature Review Run
 * |--------------------------------------------------------------------------
 */

// POST /api/review/generate - starts a review (this is where the browser's form data arrives).
//  1. Validates topic, start_year, end_year (integers, 1900-2100, start <= end) and max_papers (1-20).
//  2. Creates a 'Run' document in MongoDB with status 'queued' - its id is the run_id shown on the page.
//  3. Forwards the same data plus run_id to the Python engine (POST /api/review/generate on port 8000).
//     Python answers with a task_id, which is stored in the Run.
//  4. Starts syncPythonRunToMongo() in the background to copy progress and results into the Run.
//  5. Immediately answers 202 {run_id, task_id, status} - it does not wait for the review to finish.
// If Python cannot be reached, the Run is saved with status 'failed' and retryable=true.
//Starts a review. This is the most important door.
app.post("/api/review/generate", async (req, res) => {
  try {
    const {
      topic,
      start_year,
      end_year,
      max_papers = 6
    } = req.body;

    if (
      !topic ||
      typeof topic !== "string"
    ) {
      return res.status(400).json({
        error: "topic is required"
      });
    }

    if (
      !Number.isInteger(start_year) ||
      !Number.isInteger(end_year)
    ) {
      return res.status(400).json({
        error:
          "start_year and end_year must be integers"
      });
    }

    if (
      start_year < 1900 ||
      start_year > 2100
    ) {
      return res.status(400).json({
        error:
          "start_year must be between 1900 and 2100"
      });
    }

    if (
      end_year < 1900 ||
      end_year > 2100
    ) {
      return res.status(400).json({
        error:
          "end_year must be between 1900 and 2100"
      });
    }

    if (start_year > end_year) {
      return res.status(400).json({
        error:
          "start_year must be less than or equal to end_year"
      });
    }

    if (
      !Number.isInteger(max_papers) ||
      max_papers < 1 ||
      max_papers > MAX_PAPERS
    ) {
      return res.status(400).json({
        error:
          `max_papers must be between 1 and ${MAX_PAPERS}`
      });
    }

    const run = await Run.create({
      topic: topic.trim(),
      start_year,
      end_year,
      max_papers,
      status: "queued"
    });

    /*
     * Forward the request to the existing Python/LangGraph
     * AI service.
     */

    let pythonResponse = null;

    try {
      const response = await fetch(
        `${PYTHON_AI_URL}/api/review/generate`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            topic: topic.trim(),
            start_year,
            end_year,
            max_papers,
            run_id: run._id.toString()
          })
        }
      );

      if (response.ok) {
        pythonResponse =
          await response.json();

        if (pythonResponse.task_id) {
          run.python_task_id =
            pythonResponse.task_id;

          run.status =
            pythonResponse.status ||
            "PROCESSING";
        }
      } else {
        const errorText =
          await response.text();

        run.status = "failed";

        run.error =
          errorText ||
          `Python AI service returned HTTP ${response.status}`;

        run.reason = run.error;
        run.retryable = false;
      }
    } catch (pythonError) {
      console.warn(
        "Python AI service is not currently reachable:",
        pythonError.message
      );

      run.status = "failed";

      run.error =
        "Python AI service is not currently reachable.";

      run.reason =
        pythonError.message;

      run.retryable = true;
    }

    if (
      pythonResponse &&
      pythonResponse.task_id
    ) {
      run.status =
        pythonResponse.status ||
        "PROCESSING";
    }

    await run.save();

    /*
     * Start background synchronization.
     */

    if (run.python_task_id) {
      syncPythonRunToMongo(
        run._id.toString(),
        run.python_task_id
      ).catch((syncError) => {
        console.error(
          "Background Python/Mongo sync failed:",
          syncError.message
        );
      });
    }

    res.status(202).json({
      success: true,
      run_id: run._id.toString(),
      task_id:
        run.python_task_id || null,
      status: run.status,
      python_service_reachable:
        Boolean(pythonResponse)
    });
  } catch (error) {
    console.error(
      "Generation request error:",
      error
    );

    res.status(500).json({
      error:
        "Failed to create literature review run"
    });
  }
});

/*
 * |--------------------------------------------------------------------------
 * | Python -> MongoDB Run Synchronization
 * |--------------------------------------------------------------------------
 */

// The courier between the Python engine and MongoDB. It runs in the background after a review is started.
// Every 3 seconds (inner poll function):
//  1. Load the Run from MongoDB and ask Python for GET /api/review/status/<task_id>.
//  2. Copy Python's progress list (which agents finished and how long) into run.progress.
//  3. COMPLETED -> save the review, papers, research gaps and evaluation into the Run and stop.
//  4. RATE_LIMITED / QUOTA_EXCEEDED -> mark 'rate_limited' with the message and stop.
//  5. FAILED -> mark 'failed' with the error and stop.
//  6. Still PROCESSING (or unknown, or a temporary network error) -> wait 3 seconds and ask again.
// The browser never talks to Python; it reads these saved results from MongoDB through GET /api/review/:id.
async function syncPythonRunToMongo(
  runId,
  pythonTaskId
) {
  // One check of the loop: look at Python's status once, update MongoDB, then either stop or schedule the next check in 3 s.
  const poll = async () => {
    try {
      const run =
        await Run.findById(runId);

      if (!run) {
        console.error(
          "Sync: MongoDB run not found:",
          runId
        );

        return;
      }

      const response =
        await fetch(
          `${PYTHON_AI_URL}/api/review/status/${pythonTaskId}`
        );

      if (!response.ok) {
        console.warn(
          "Sync: Python status request failed:",
          response.status
        );

        setTimeout(
          poll,
          3000
        );

        return;
      }

      const result =
        await response.json();

      console.log(
        `Sync: Python task ${pythonTaskId} -> ${result.status}`
      );

      // Mirror the per-agent progress reported by the AI engine.
      if (
        Array.isArray(result.progress) &&
        result.progress.length !== (run.progress || []).length
      ) {
        run.progress = result.progress;
        run.markModified("progress");
        await run.save();
      }

      /*
       * ---------------------------------------------------------------
       * COMPLETED
       * ---------------------------------------------------------------
       */

      if (
        result.status === "COMPLETED"
      ) {
        const data =
          result.data || {};

        run.status =
          "completed";

        run.data =
          data;

        run.final_review =
          data.final_review_md ||
          data.final_review ||
          "";

        run.final_review_md =
          run.final_review;

        run.raw_papers_list =
          data.raw_papers_list ||
          data.papers ||
          [];

        run.papers =
          run.raw_papers_list;

        run.research_gap =
          data.research_gap ||
          data.research_gap_analysis ||
          null;

        run.evaluation =
          result.evaluation ||
          data.evaluation ||
          null;

        run.error = undefined;
        run.reason = undefined;
        run.retryable = false;

        await run.save();

        console.log(
          `Sync: MongoDB run ${runId} marked COMPLETED.`
        );

        return;
      }

      /*
       * ---------------------------------------------------------------
       * RATE LIMITED / GROQ QUOTA
       * ---------------------------------------------------------------
       */

      if (
        result.status ===
          "RATE_LIMITED" ||
        result.status ===
          "QUOTA_EXCEEDED"
      ) {
        run.status =
          "rate_limited";

        run.error =
          result.error ||
          result.reason ||
          "Groq daily token limit reached. Generation halted.";

        run.reason =
          result.reason ||
          run.error;

        run.retryable =
          typeof result.retryable ===
          "boolean"
            ? result.retryable
            : true;

        await run.save();

        console.log(
          `Sync: MongoDB run ${runId} marked RATE_LIMITED.`
        );

        return;
      }

      /*
       * ---------------------------------------------------------------
       * FAILED
       * ---------------------------------------------------------------
       */

      if (
        result.status ===
        "FAILED"
      ) {
        run.status =
          "failed";

        run.error =
          result.error ||
          "The multi-agent literature review workflow failed.";

        run.reason =
          result.reason ||
          run.error;

        run.retryable = false;

        await run.save();

        console.log(
          `Sync: MongoDB run ${runId} marked FAILED.`
        );

        return;
      }

      /*
       * ---------------------------------------------------------------
       * STILL RUNNING
       * ---------------------------------------------------------------
       */

      if (
        [
          "PROCESSING",
          "RUNNING",
          "STARTED",
          "PENDING",
          "QUEUED"
        ].includes(
          result.status
        )
      ) {
        setTimeout(
          poll,
          3000
        );

        return;
      }

      /*
       * Unknown status:
       * continue polling rather than prematurely failing.
       */

      setTimeout(
        poll,
        3000
      );
    } catch (error) {
      console.error(
        "Sync error:",
        error.message
      );

      /*
       * Temporary communication errors should not destroy
       * the literature review run.
       */

      setTimeout(
        poll,
        3000
      );
    }
  };

  await poll();
}

/*
 * |--------------------------------------------------------------------------
 * | Human-in-the-loop feedback on a finished review
 * |--------------------------------------------------------------------------
 */

// POST /api/review/:id/feedback - human-in-the-loop feedback on a finished review.
//  1. Reads the reviewer's comment (at least 3 characters) and finds the Run and its python_task_id.
//  2. Sends the comment to Python (POST /api/review/feedback/<task_id>), where the Reviser rewrites the review.
//  3. Saves a short record in run.feedback_history: the comment, the time and the critic scores before and after.
//  4. Replaces run.final_review_md (and run.data.final_review_md) with the revised review and updates run.evaluation.
//  5. Returns the revised review and the scores to the browser.
// Sends the reviewer’s comment to Python and saves the new review.
app.post("/api/review/:id/feedback", async (req, res) => {
  try {
    const comments = String(req.body?.comments || "").trim();
    if (comments.length < 3) {
      return res.status(400).json({ error: "Please write some feedback first." });
    }

    const run = mongoose.isValidObjectId(req.params.id) ? await Run.findById(req.params.id) : null;
    if (!run || !run.python_task_id) {
      return res.status(404).json({ error: "Run not found" });
    }

    const response = await fetch(
      `${PYTHON_AI_URL}/api/review/feedback/${run.python_task_id}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      }
    );
    const payload = await response.json();

    if (!response.ok) {
      return res.status(response.status).json({
        error: payload.detail || "Feedback revision failed"
      });
    }

    // keep a compact record of every human feedback round with the critic scores before and after
    // Shrinks a long critic report to the few numbers worth saving: score, supported-claims fraction,
    // paper coverage and the number of open issues.
    const summary = (critic = {}) => ({
      score: critic.score,
      supported_fraction: critic.stats?.supported_fraction,
      paper_coverage: critic.stats?.paper_coverage,
      open_issues: Array.isArray(critic.issues) ? critic.issues.length : undefined
    });
    run.feedback_history = [
      ...(run.feedback_history || []),
      {
        comments,
        submitted_at: new Date().toISOString(),
        critic_before: summary(payload.critic_before),
        critic_after: summary(payload.critic_after)
      }
    ];
    run.markModified("feedback_history");

    run.final_review_md = payload.final_review_md;
    run.evaluation = payload.evaluation;
    if (run.data) {
      run.data = { ...run.data, final_review_md: payload.final_review_md };
      run.markModified("data");
    }
    await run.save();

    res.json({ success: true, ...payload });
  } catch (error) {
    console.error("Feedback error:", error);
    res.status(500).json({ error: "Failed to apply feedback" });
  }
});

/*
 * |--------------------------------------------------------------------------
 * | Get Run
 * |--------------------------------------------------------------------------
 */

// GET /api/review/:id - the endpoint the browser polls every 3 seconds to read a run.
// Checks the id format, loads the Run from MongoDB and returns it. If top-level fields are empty it fills them
// from run.data (final_review_md, raw_papers_list/papers and research_gap), so the page can always find them.
// Returns 404 when the run does not exist.
app.get(
  "/api/review/:id",
  async (req, res) => {
    try {
      if (!mongoose.isValidObjectId(req.params.id)) {
        return res.status(404).json({
          error: "Run not found"
        });
      }

      const run =
        await Run.findById(
          req.params.id
        );

      if (!run) {
        return res.status(404).json({
          error: "Run not found"
        });
      }

      const runObj = run.toObject ? run.toObject() : { ...run };
      if (runObj.data) {
        if (!runObj.final_review_md && runObj.data.final_review_md) {
          runObj.final_review_md = runObj.data.final_review_md;
        }
        if (!runObj.final_review && runObj.data.final_review) {
          runObj.final_review = runObj.data.final_review;
        }
        if ((!runObj.raw_papers_list || runObj.raw_papers_list.length === 0) && runObj.data.raw_papers_list) {
          runObj.raw_papers_list = runObj.data.raw_papers_list;
        }
        if ((!runObj.papers || runObj.papers.length === 0) && (runObj.raw_papers_list || runObj.data.papers)) {
          runObj.papers = runObj.raw_papers_list || runObj.data.papers;
        }
        if (!runObj.research_gap && (runObj.data.research_gap || runObj.data.research_gap_analysis)) {
          runObj.research_gap = runObj.data.research_gap || runObj.data.research_gap_analysis;
        }
      }

      res.json(runObj);
    } catch (error) {
      console.error(
        "Run lookup error:",
        error
      );

      res.status(500).json({
        error:
          "Failed to retrieve run"
      });
    }
  }
);

/*
 * |--------------------------------------------------------------------------
 * | MongoDB Connection
 * |--------------------------------------------------------------------------
 */

// Starts the Node server: first connects to MongoDB (MONGODB_URI); only if that works does it start listening
// on the PORT (default 3000). If MongoDB cannot be reached it prints the error and stops the program.
async function startServer() {
  try {
    await mongoose.connect(
      MONGODB_URI
    );

    console.log(
      "MongoDB connected"
    );

    console.log(
      `Database: ${MONGODB_URI}`
    );

    app.listen(
      PORT,
      () => {
        console.log(
          `Node/Express backend running at http://127.0.0.1:${PORT}`
        );
      }
    );
  } catch (error) {
    console.error(
      "MongoDB connection failed:"
    );

    console.error(
      error.message
    );

    process.exit(1);
  }
}

startServer();