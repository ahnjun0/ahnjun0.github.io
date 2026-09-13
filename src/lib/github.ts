// 빌드 시 GitHub 저장소 메타데이터를 가져온다. 실패(private/404/레이트리밋)하면 null.
// Actions에서는 GITHUB_TOKEN이 자동 주입되고, 로컬에서는 없으면 비인증(60 req/h)으로 시도한다.
export type RepoMeta = { stars: number; pushedAt: string; language: string | null; url: string };

const cache = new Map<string, Promise<RepoMeta | null>>();

export function repoMeta(repoUrl: string): Promise<RepoMeta | null> {
  const m = repoUrl.match(/github\.com\/([^/]+\/[^/#?]+)/);
  if (!m) return Promise.resolve(null);
  const full = m[1].replace(/\.git$/, "");
  if (!cache.has(full)) cache.set(full, fetchMeta(full));
  return cache.get(full)!;
}

async function fetchMeta(full: string): Promise<RepoMeta | null> {
  const token = process.env.GITHUB_TOKEN ?? process.env.GH_TOKEN;
  try {
    const res = await fetch(`https://api.github.com/repos/${full}`, {
      headers: {
        Accept: "application/vnd.github+json",
        "User-Agent": "ahnjun0.github.io-build",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    });
    if (!res.ok) return null;
    const j = await res.json();
    if (j.private) return null;
    return { stars: j.stargazers_count ?? 0, pushedAt: (j.pushed_at ?? "").slice(0, 10).replace(/-/g, "."), language: j.language ?? null, url: j.html_url };
  } catch {
    return null;
  }
}

export async function publicRepoCount(user: string): Promise<number | null> {
  const token = process.env.GITHUB_TOKEN ?? process.env.GH_TOKEN;
  try {
    const res = await fetch(`https://api.github.com/users/${user}`, {
      headers: { Accept: "application/vnd.github+json", "User-Agent": "ahnjun0.github.io-build", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    });
    if (!res.ok) return null;
    return (await res.json()).public_repos ?? null;
  } catch { return null; }
}
