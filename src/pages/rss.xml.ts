import type { APIContext } from "astro";
import { getCollection } from "astro:content";
import { site } from "../config";

export async function GET(ctx: APIContext) {
  const notes = (await getCollection("notes", ({ data }) => !data.draft))
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
  const base = ctx.site?.toString().replace(/\/$/, "") ?? "";
  const esc = (s: string) => s.replace(/[<>&]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" })[c]!);
  const items = notes.map((n) => `<item><title>${esc(n.data.title)}</title><link>${base}/notes/${n.id}</link><guid>${base}/notes/${n.id}</guid><pubDate>${n.data.date.toUTCString()}</pubDate>${n.data.description ? `<description>${esc(n.data.description)}</description>` : ""}</item>`).join("");
  const xml = `<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>${esc(site.nameEn)} — Notes</title><link>${base}</link><description>${esc(site.description)}</description>${items}</channel></rss>`;
  return new Response(xml, { headers: { "Content-Type": "application/xml; charset=utf-8" } });
}
