const escapedLineBreak = String.raw`\n`;
const escapedWindowsLineBreak = String.raw`\r\n`;

function isJsonDocument(text: string): boolean {
  const trimmed = text.trim();
  if (!(trimmed.startsWith("{") || trimmed.startsWith("["))) return false;
  try {
    JSON.parse(trimmed);
    return true;
  } catch {
    return false;
  }
}

function hasEscapedMarkdownStructure(text: string): boolean {
  const lineBreak = String.raw`(?:\\r\\n|\\n)`;
  return new RegExp(
    `${lineBreak}${lineBreak}|${lineBreak}[ \\t]*(?:[-+*][ \\t]+|\\d+[.)][ \\t]+|#{1,6}[ \\t]+|>)`,
  ).test(text);
}

/**
 * Recover Markdown structure from model replies that escaped every line break.
 *
 * This is deliberately conservative: only fully escaped, visibly structured
 * prose is repaired. JSON documents, fenced code, ordinary string escapes,
 * mixed real/escaped line endings, and LaTeX commands remain byte-for-byte
 * unchanged.
 */
export function normalizeEscapedMarkdownLineBreaks(text: string): string {
  if (
    (!text.includes(escapedLineBreak) && !text.includes(escapedWindowsLineBreak))
    || text.includes("\n")
    || text.includes("\r")
    || text.includes("```")
    || text.includes("~~~")
    || isJsonDocument(text)
    || !hasEscapedMarkdownStructure(text)
  ) {
    return text;
  }

  return text
    .replaceAll(`${escapedWindowsLineBreak}${escapedWindowsLineBreak}`, "\n\n")
    .replaceAll(`${escapedLineBreak}${escapedLineBreak}`, "\n\n")
    .replace(
      /(?:\\r\\n|\\n)(?=[ \t]*(?:[-+*](?:[ \t]+|$)|\d+[.)](?:[ \t]+|$)|#{1,6}(?:[ \t]+|$)|>|$))/g,
      "\n",
    );
}
