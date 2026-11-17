import React from 'react';
import { BookOpen } from 'lucide-react';

/**
 * Checks if a string is a raw technical chunk identifier
 */
function isTechnicalId(str) {
  if (!str) return true;
  return /^(counselchat_chunk_|chunk_|[a-zA-Z0-9_\-]+__chunk|[a-zA-Z0-9_\-]+:\d+)/i.test(str.trim());
}

/**
 * Resolves a raw chunk ID to a user-friendly citation label
 */
function resolveCitationLabel(raw, citations = []) {
  const clean = raw.replace(/^(?:citation|source|doc|chunk)\s*:\s*/i, '').trim();
  const lower = clean.toLowerCase();
  const isCounselChat = lower.includes('counselchat') || lower.includes('counsel');

  const matched = citations.find((c) => {
    if (!c) return false;
    const docId = (c.document_id || '').toLowerCase();
    const chunkId = (c.chunk_id || '').toLowerCase();
    const title = (c.title || '').toLowerCase();
    return (
      (docId && (docId === lower || lower.includes(docId) || docId.includes(lower))) ||
      (chunkId && (chunkId === lower || lower.includes(chunkId) || chunkId.includes(lower))) ||
      (title && title === lower)
    );
  });

  if (matched && matched.title && !isTechnicalId(matched.title)) {
    const title = matched.title.length > 40 ? matched.title.substring(0, 37) + '...' : matched.title;
    return isCounselChat ? `[Community Perspective: ${title}]` : `[Source: ${title}]`;
  }

  return isCounselChat ? '[Community Counseling Reference]' : '[Verified Evidence Reference]';
}

/**
 * Replaces raw technical chunk identifiers with clean citation labels
 */
export function cleanInlineCitations(text, citations = []) {
  if (!text) return '';

  // 1. Bracketed tags: [counselchat_chunk_...], [citation: ...], [somatic_regulation__chunk_0]
  let result = text.replace(
    /\[((?:citation|source|doc|chunk)?\s*:?\s*[a-zA-Z0-9_\-:]*(?:chunk|counselchat|regulation)[a-zA-Z0-9_\-:]*)\]/gi,
    (match, p1) => resolveCitationLabel(p1, citations)
  );

  // 2. Parenthesized tags: (counselchat_chunk_...), (citation: ...)
  result = result.replace(
    /\(((?:citation|source|doc|chunk)?\s*:?\s*[a-zA-Z0-9_\-:]*(?:chunk|counselchat|regulation)[a-zA-Z0-9_\-:]*)\)/gi,
    (match, p1) => resolveCitationLabel(p1, citations)
  );

  // 3. Standalone unbracketed identifiers: counselchat_chunk_..., ...__chunk_\d+, chunk_...
  result = result.replace(
    /\b(counselchat_chunk_[a-zA-Z0-9_\-]+|[a-zA-Z0-9_\-]+__chunk_[a-zA-Z0-9_\-]+|chunk_[a-zA-Z0-9_\-]+)\b/gi,
    (match, p1) => resolveCitationLabel(p1, citations)
  );

  return result;
}

/**
 * Lightweight, safe markdown formatter for MindBridge responses.
 * Formats:
 * - ### Headings
 * - **bold** and *italic*
 * - Unordered lists (- item)
 * - Numbered lists (1. item)
 * - Blockquotes and alerts (> [!IMPORTANT])
 * - Links [label](url)
 * - Clean evidence citation badges
 */
export function MarkdownView({ content, citations = [], className = '' }) {
  if (!content) return null;

  const sanitizedContent = cleanInlineCitations(content, citations);
  const lines = sanitizedContent.split('\n');
  const elements = [];
  let currentList = null;

  lines.forEach((line, index) => {
    const trimmed = line.trim();

    // Empty line
    if (!trimmed) {
      if (currentList) {
        elements.push(renderList(currentList, `list-${index}`));
        currentList = null;
      }
      return;
    }

    // List item (e.g. - item or * item)
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      const itemText = trimmed.slice(2);
      if (!currentList || currentList.type !== 'ul') {
        if (currentList) elements.push(renderList(currentList, `list-prev-${index}`));
        currentList = { type: 'ul', items: [] };
      }
      currentList.items.push(itemText);
      return;
    }

    // Numbered list item
    const numberedMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);
    if (numberedMatch) {
      const itemText = numberedMatch[2];
      if (!currentList || currentList.type !== 'ol') {
        if (currentList) elements.push(renderList(currentList, `list-prev-${index}`));
        currentList = { type: 'ol', items: [] };
      }
      currentList.items.push(itemText);
      return;
    }

    // Flush active list if line is not a list item
    if (currentList) {
      elements.push(renderList(currentList, `list-${index}`));
      currentList = null;
    }

    // Alert or Blockquote
    if (trimmed.startsWith('>')) {
      const quoteText = trimmed.replace(/^>\s*/, '');
      const isImportant = quoteText.includes('[!IMPORTANT]') || quoteText.includes('Direct Calling Required');
      elements.push(
        <div
          key={`quote-${index}`}
          className={`my-3 p-3.5 rounded-xl text-sm leading-relaxed ${
            isImportant
              ? 'bg-tertiary-fixed/30 border-l-4 border-tertiary text-on-surface'
              : 'bg-surface-container border-l-4 border-secondary text-on-surface-variant'
          }`}
        >
          {parseInlineFormatting(quoteText.replace(/\[!IMPORTANT\]/g, 'Important:'))}
        </div>
      );
      return;
    }

    // Headings
    if (trimmed.startsWith('### ')) {
      elements.push(
        <h3 key={`h3-${index}`} className="font-headline-sm text-headline-sm text-primary font-semibold mt-4 mb-2">
          {parseInlineFormatting(trimmed.slice(4))}
        </h3>
      );
      return;
    }
    if (trimmed.startsWith('## ')) {
      elements.push(
        <h2 key={`h2-${index}`} className="font-headline-md text-headline-md text-primary font-semibold mt-5 mb-2">
          {parseInlineFormatting(trimmed.slice(3))}
        </h2>
      );
      return;
    }
    if (trimmed.startsWith('# ')) {
      elements.push(
        <h1 key={`h1-${index}`} className="font-display-mobile text-display-mobile text-primary font-bold mt-6 mb-3">
          {parseInlineFormatting(trimmed.slice(2))}
        </h1>
      );
      return;
    }

    // Standard paragraph
    elements.push(
      <p key={`p-${index}`} className="mb-2.5 leading-relaxed font-body-md text-body-md">
        {parseInlineFormatting(trimmed)}
      </p>
    );
  });

  if (currentList) {
    elements.push(renderList(currentList, 'list-final'));
  }

  return <div className={`prose-sm max-w-none text-on-surface ${className}`}>{elements}</div>;
}

function renderList(list, key) {
  const ListTag = list.type === 'ol' ? 'ol' : 'ul';
  return (
    <ListTag
      key={key}
      className={`my-2 space-y-1.5 pl-5 ${list.type === 'ol' ? 'list-decimal' : 'list-disc text-secondary'}`}
    >
      {list.items.map((item, i) => (
        <li key={i} className="text-on-surface font-body-md text-body-md pl-1">
          {parseInlineFormatting(item)}
        </li>
      ))}
    </ListTag>
  );
}

function parseInlineFormatting(text) {
  if (!text) return '';

  const parts = [];
  let remaining = text;
  let keyIdx = 0;

  // Regex to catch **bold**, *italic*, [link](url), and [citation badges]
  const pattern = /(\*\*([^*]+)\*\*|\*([^*]+)\*|\[([^\]]+)\]\(([^)]+)\)|\[(Community Counseling Reference|Verified Evidence Reference|Source:\s*[^\]]+|Community Perspective:\s*[^\]]+)\])/;

  while (remaining) {
    const match = remaining.match(pattern);
    if (!match) {
      parts.push(remaining);
      break;
    }

    const matchStart = match.index;
    if (matchStart > 0) {
      parts.push(remaining.substring(0, matchStart));
    }

    const fullMatch = match[0];
    if (fullMatch.startsWith('**')) {
      // Bold
      parts.push(
        <strong key={keyIdx++} className="font-semibold text-on-surface">
          {match[2]}
        </strong>
      );
    } else if (fullMatch.startsWith('*')) {
      // Italic
      parts.push(
        <em key={keyIdx++} className="italic text-on-surface-variant">
          {match[3]}
        </em>
      );
    } else if (fullMatch.startsWith('[') && fullMatch.includes('](')) {
      // Link [label](url)
      const label = match[4];
      const url = match[5];
      parts.push(
        <a
          key={keyIdx++}
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary underline hover:text-primary-container font-medium transition-colors"
        >
          {label}
        </a>
      );
    } else if (fullMatch.startsWith('[') && fullMatch.endsWith(']')) {
      // Citation badge
      const badgeText = fullMatch.slice(1, -1);
      parts.push(
        <span
          key={keyIdx++}
          className="inline-flex items-center gap-1 px-1.5 py-0.5 mx-0.5 rounded-md bg-secondary-container/50 text-secondary font-label-sm text-[11px] font-medium align-baseline border border-secondary/20 shadow-2xs select-none"
          title={`Evidence Reference: ${badgeText}`}
        >
          <BookOpen className="w-3 h-3 text-secondary shrink-0" />
          <span>{badgeText}</span>
        </span>
      );
    }

    remaining = remaining.substring(matchStart + fullMatch.length);
  }

  return parts;
}
