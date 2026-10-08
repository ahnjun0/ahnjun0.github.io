// @ts-check
import { defineConfig } from "astro/config";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeTableA11y from "./src/lib/rehype-table-a11y.mjs";

export default defineConfig({
  site: "https://ahnjun0.github.io",
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex, rehypeTableA11y],
    shikiConfig: { theme: "github-light-high-contrast", wrap: true },
  },
});
