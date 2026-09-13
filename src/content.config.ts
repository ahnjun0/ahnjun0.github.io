import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const projects = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/projects" }),
  schema: z.object({
    title: z.string(),
    summary: z.string(),            // 카드/목록 한 줄
    period: z.string(),             // "2026.08 – 09"
    sortDate: z.string(),           // "2026-08-01" 정렬용
    category: z.enum(["AI/ML", "Data", "Web/App", "Game AI", "Infra", "Product"]),
    role: z.string().optional(),    // "개인" | "팀 · 기획/분석"
    result: z.string().optional(),  // "트랙 3등 · 원장상"
    tags: z.array(z.string()).default([]),
    featured: z.number().optional(),// 홈 노출 순서 (1~4)
    repo: z.string().url().optional(),
    link: z.string().url().optional(),
    draft: z.boolean().default(false),
  }),
});

const notes = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/notes" }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    date: z.coerce.date(),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
  }),
});

export const collections = { projects, notes };
