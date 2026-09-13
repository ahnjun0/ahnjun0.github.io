// 언어 유틸. 경로 앞에 /en 이 붙으면 영어.
export type Lang = "ko" | "en";
export const langs: Lang[] = ["ko", "en"];

export function langFromUrl(url: URL): Lang {
  return url.pathname === "/en" || url.pathname.startsWith("/en/") ? "en" : "ko";
}
export function prefix(lang: Lang) { return lang === "en" ? "/en" : ""; }
/** 같은 페이지의 다른 언어 경로 */
export function switchPath(pathname: string, to: Lang): string {
  const bare = pathname.replace(/^\/en(?=\/|$)/, "") || "/";
  return to === "en" ? (bare === "/" ? "/en/" : `/en${bare}`) : bare;
}

export const ui = {
  ko: {
    nav: { about: "About", projects: "Projects", experience: "Experience", awards: "Awards", notes: "Notes", contact: "Contact" },
    hero: { email: "Email", github: "GitHub", cv: "CV" },
    sections: { about: "About", projects: "Projects", experience: "Experience", awards: "Awards", contact: "Contact" },
    projectsSub: (f: number, a: number) => `대표 ${f}개 · 전체 ${a}개`,
    awardsSub: (n: number) => `${n}건`,
    allProjects: "전체 프로젝트 보기 →",
    contactLead: "이야기 나누고 싶다면,",
    contactSub: (repos: number | null) => `GitHub @ahnjun0${repos !== null ? ` (public 저장소 ${repos}개)` : ""} · solved.ac ahnjun0 · 아마추어무선 6K5EHL`,
    projectsIntro: "대회, 캠프, 동아리, 혼자 만든 것까지. 각 페이지는 <b>문제 → 접근 → 결과 → 배운 점</b> 순서로 씁니다.",
    filterAll: "전체",
    notesIntro: "공부하며 남긴 기록. 2023년 퍼셉트론 노트부터 그대로 둡니다.",
    notesEnOnly: "",
    backProjects: "← Projects", backNotes: "← Notes",
    facts: { role: "ROLE", result: "RESULT", stack: "STACK", links: "LINKS", github: "GITHUB", repo: "Repository", site: "Site", lastCommit: "마지막 커밋" },
    notFound: "여기엔 아무것도 없습니다.", home: "홈으로 →",
  },
  en: {
    nav: { about: "About", projects: "Projects", experience: "Experience", awards: "Awards", notes: "Notes", contact: "Contact" },
    hero: { email: "Email", github: "GitHub", cv: "CV" },
    sections: { about: "About", projects: "Projects", experience: "Experience", awards: "Awards", contact: "Contact" },
    projectsSub: (f: number, a: number) => `${f} featured · ${a} total`,
    awardsSub: (n: number) => `${n}`,
    allProjects: "All projects →",
    contactLead: "If you'd like to talk,",
    contactSub: (repos: number | null) => `GitHub @ahnjun0${repos !== null ? ` (${repos} public repos)` : ""} · solved.ac ahnjun0 · Amateur radio 6K5EHL`,
    projectsIntro: "Competitions, camps, club work, and solo builds. Each page follows <b>problem → approach → result → what I learned</b>.",
    filterAll: "All",
    notesIntro: "Study notes, kept as written. These are in Korean.",
    notesEnOnly: "Notes are written in Korean only.",
    backProjects: "← Projects", backNotes: "← Notes",
    facts: { role: "ROLE", result: "RESULT", stack: "STACK", links: "LINKS", github: "GITHUB", repo: "Repository", site: "Site", lastCommit: "last commit" },
    notFound: "There's nothing here.", home: "Home →",
  },
} as const;
