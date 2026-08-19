import DOMPurify from "dompurify";
import renderMathInElement from "katex/contrib/auto-render";
import { marked } from "marked";
import { useEffect, useMemo, useRef } from "react";
import { normalizeEscapedMarkdownLineBreaks } from "./markdownText";

interface MarkdownProps {
  text: string;
  live?: boolean;
  normalizeEscapedLineBreaks?: boolean;
}

export function Markdown({
  text,
  live = false,
  normalizeEscapedLineBreaks = false,
}: MarkdownProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  const markdown = useMemo(
    () => normalizeEscapedLineBreaks ? normalizeEscapedMarkdownLineBreaks(text) : text,
    [normalizeEscapedLineBreaks, text],
  );
  const html = useMemo(
    () =>
      DOMPurify.sanitize(
        marked.parse(markdown, {
          breaks: true,
          gfm: true,
        }) as string,
      ),
    [markdown],
  );

  useEffect(() => {
    if (live || !rootRef.current) return;
    renderMathInElement(rootRef.current, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
        { left: "$", right: "$", display: false },
      ],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
      strict: "ignore",
      throwOnError: false,
      trust: false,
    });
    const buttons: HTMLButtonElement[] = [];
    for (const block of rootRef.current.querySelectorAll("pre")) {
      const source = block.querySelector("code")?.textContent || block.textContent || "";
      const button = document.createElement("button");
      button.className = "code-copy";
      button.type = "button";
      button.textContent = "Copy";
      button.addEventListener("click", () => {
        void navigator.clipboard.writeText(source).then(() => {
          button.textContent = "Copied";
          window.setTimeout(() => {
            button.textContent = "Copy";
          }, 1200);
        });
      });
      block.append(button);
      buttons.push(button);
    }
    return () => buttons.forEach((button) => button.remove());
  }, [html, live]);

  return <div className="rich-text" dangerouslySetInnerHTML={{ __html: html }} ref={rootRef} />;
}
