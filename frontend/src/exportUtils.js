import { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType } from 'docx';
import { saveAs } from 'file-saver';

/**
 * Clean markdown text from bold/italic markers for docx text runs
 */
// Turns one line of Markdown text into Word text pieces: **bold**, *italic*, `code` and [1] citation marks each
// become a styled run (Calibri, size 11) so the Word file looks like the manuscript.
function parseInlineFormatting(text) {
  if (!text) return [];
  
  const parts = [];
  const regex = /(\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`|\[(\d+)\])/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(new TextRun({ text: text.slice(lastIndex, match.index), font: 'Calibri', size: 22 }));
    }
    
    if (match[2]) {
      // Bold
      parts.push(new TextRun({ text: match[2], bold: true, font: 'Calibri', size: 22 }));
    } else if (match[3]) {
      // Italic
      parts.push(new TextRun({ text: match[3], italics: true, font: 'Calibri', size: 22 }));
    } else if (match[4]) {
      // Code
      parts.push(new TextRun({ text: match[4], font: 'Consolas', size: 20, color: '334155' }));
    } else if (match[5]) {
      // In-text Citation [i]
      parts.push(new TextRun({ text: `[${match[5]}]`, bold: true, color: '2563EB', font: 'Calibri', size: 22 }));
    }
    
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    parts.push(new TextRun({ text: text.slice(lastIndex), font: 'Calibri', size: 22 }));
  }

  return parts.length > 0 ? parts : [new TextRun({ text, font: 'Calibri', size: 22 })];
}

/**
 * Export Literature Review to Microsoft Word (.docx)
 */
// Downloads the manuscript as a Word (.docx) file.
// It builds the document title, then converts the Markdown headings, paragraphs and lists line by line,
// adds the reviewed papers, and saves the file. If there is no content it shows an alert instead.
export async function exportToWord(markdownText, topic, papers = []) {
  if (!markdownText && papers.length === 0) {
    alert('No manuscript content available to export.');
    return;
  }

  const paragraphs = [];

  // Document Title
  paragraphs.push(
    new Paragraph({
      text: `Literature Review: ${topic || 'Academic Research Synthesis'}`,
      heading: HeadingLevel.TITLE,
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 }
    })
  );

  // Subtitle / Metadata
  const dateStr = new Date().toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });

  paragraphs.push(
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 400 },
      children: [
        new TextRun({ text: `Generated on ${dateStr} • Multi-Agent Scholarly Literature Generator\n`, italics: true, color: '64748B', size: 20 }),
        new TextRun({ text: `Synthesized Evidence Corpus: ${papers.length} Peer-Reviewed Studies`, bold: true, color: '475569', size: 20 })
      ]
    })
  );

  // Parse lines of markdown text
  const rawLines = (markdownText || '').split('\n');

  for (let i = 0; i < rawLines.length; i++) {
    const line = rawLines[i].trim();
    if (!line) continue;

    // Skip redundant matrix tables if any
    if (line.startsWith('|') || line.toLowerCase().includes('comparison matrix')) {
      continue;
    }

    if (line.startsWith('# ')) {
      const headingText = line.replace(/^#\s+/, '').trim();
      paragraphs.push(
        new Paragraph({
          text: headingText,
          heading: HeadingLevel.HEADING_1,
          spacing: { before: 300, after: 150 }
        })
      );
    } else if (line.startsWith('## ')) {
      const headingText = line.replace(/^##\s+/, '').trim();
      paragraphs.push(
        new Paragraph({
          text: headingText,
          heading: HeadingLevel.HEADING_2,
          spacing: { before: 250, after: 120 }
        })
      );
    } else if (line.startsWith('### ')) {
      const headingText = line.replace(/^###\s+/, '').trim();
      paragraphs.push(
        new Paragraph({
          text: headingText,
          heading: HeadingLevel.HEADING_3,
          spacing: { before: 200, after: 100 }
        })
      );
    } else if (line.startsWith('- ') || line.startsWith('* ')) {
      const bulletText = line.replace(/^[-*]\s+/, '').trim();
      paragraphs.push(
        new Paragraph({
          children: parseInlineFormatting(bulletText),
          bullet: { level: 0 },
          spacing: { after: 100 }
        })
      );
    } else if (/^\[\d+\]/.test(line)) {
      // Clean reference entry: remove any duplicate like [1] [1]
      const cleanRef = line.replace(/^\[(\d+)\]\s*\[\1\]/, '[$1]').replace(/^\d+\.\s*\[(\d+)\]/, '[$1]');
      paragraphs.push(
        new Paragraph({
          children: parseInlineFormatting(cleanRef),
          spacing: { after: 120 }
        })
      );
    } else {
      // Normal paragraph
      paragraphs.push(
        new Paragraph({
          children: parseInlineFormatting(line),
          spacing: { after: 180, line: 276 } // 1.15 line spacing
        })
      );
    }
  }

  // Create docx document
  const doc = new Document({
    sections: [
      {
        properties: {
          page: {
            margin: {
              top: 1440, // 1 inch = 1440 twips
              right: 1440,
              bottom: 1440,
              left: 1440
            }
          }
        },
        children: paragraphs
      }
    ]
  });

  const blob = await Packer.toBlob(doc);
  const cleanFilename = (topic || 'Literature_Review')
    .replace(/[^a-zA-Z0-9_-]/g, '_')
    .slice(0, 50);
  saveAs(blob, `${cleanFilename}_Manuscript.docx`);
}

/**
 * Print / Save as PDF
 */
// Exports to PDF by opening the browser's print dialog (the user chooses 'Save as PDF').
export function exportToPdf() {
  window.print();
}

/**
 * Download Markdown file
 */
// Downloads the manuscript as a .md file. The file name is made from the topic (safe characters, max 50).
export function exportToMarkdown(markdownText, topic) {
  const blob = new Blob([markdownText || ''], { type: 'text/markdown;charset=utf-8' });
  const cleanFilename = (topic || 'Literature_Review')
    .replace(/[^a-zA-Z0-9_-]/g, '_')
    .slice(0, 50);
  saveAs(blob, `${cleanFilename}_Manuscript.md`);
}

/**
 * Copy to clipboard
 */
// Copies the text to the clipboard. Returns true if it worked and false if the browser refused.
export async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text || '');
    return true;
  } catch (err) {
    console.error('Failed to copy text: ', err);
    return false;
  }
}
