// 사이트 전체에서 쓰는 정적 콘텐츠. 프로젝트/노트는 src/content/ 의 마크다운으로 관리.
export const site = {
  name: "안준영",
  nameEn: "Junyeong Ahn",
  tagline: "문제를 정의하고, 데이터로 풀고, 서비스로 만듭니다",
  description:
    "안준영(Junyeong Ahn) 포트폴리오. 부산대학교 정보컴퓨터공학부. LLM 파인튜닝, 공공데이터 분석, 게임 AI, 브라우저 확장까지 — 모델과 배포 사이 어디든.",
  email: "jyahn.it@gmail.com",
  github: "https://github.com/ahnjun0",
  solvedac: "https://solved.ac/ahnjun0",
  callsign: "6K5EHL",
  location: "Busan, Korea",
  cv: "", // CV PDF가 준비되면 "/files/cv.pdf" 로. 비어 있으면 링크 숨김
};

export const hero = {
  greeting: "안녕하세요 👋",
  lead: "문제를 정의하고, 데이터로 풀고, 서비스로 만듭니다.",
  sub: "부산대학교 정보컴퓨터공학부 3학년. AI와 풀스택 사이에서 일합니다.",
};

export const about = {
  paragraphs: [
    "**LLM 파인튜닝**부터 **공공데이터 분석**, **게임 AI**, **브라우저 확장**까지. 모델 한 층에 머무르기보다, 문제를 정의하는 일부터 배포해서 쓰이게 하는 일까지 전부 해보는 쪽을 택해 왔습니다.",
    "2022년 학부 AI 대회 금상 두 개로 시작했고, 군에서 보안관제·체계관제를 맡았습니다. 2026년 복학 후 KAIST 몰입캠프, AID 회장, 카카오테크캠퍼스를 연달아 지나왔고, 지금은 제5회 대학 연합 딥러닝 챌린지 2026의 최종 결과를 기다리고 있습니다.",
  ],
  skills: ["Python", "PyTorch", "LoRA / vLLM", "Django", "FastAPI", "TypeScript", "Docker", "LiveKit", "Chrome Extension"],
};

export type Experience = {
  title: string; org: string; range: string; now?: boolean; bullets?: string[]; link?: string;
};
export const experience: Experience[] = [
  {
    title: "카카오테크캠퍼스 4기 · IRYA", org: "인프라 · 백엔드", range: "2026.05 – 현재", now: true, link: "https://github.com/kakaotechcampus-4/ktc4-pusan-1",
    bullets: ["AI 화상면접 지원 서비스의 LiveKit 자체 호스팅, Caddy HTTPS, 배포 파이프라인 담당"],
  },
  {
    title: "AID 부산대 AI 동아리", org: "회장", range: "2026.03 – 2026.08", link: "https://github.com/PNU-AID",
    bullets: [
      "회원 40여 명 운영, 세미나 기획·발표, 연구소 견학·홈커밍데이 주최",
      "공식 웹사이트와 대회 자동채점·리더보드 시스템 구축 (Django, PostgreSQL, Docker)",
    ],
  },
  {
    title: "KAIST 몰입캠프", org: "4주 4프로젝트", range: "2026.01 – 2026.02",
    bullets: [
      "CallCops — 전화망(8kHz) 실시간 오디오 워터마킹 모델 설계·학습",
      "Roomie 프론트엔드, KaHook! (Three.js 파티 게임), Momento (React Native + NestJS)",
    ],
  },
  { title: "대한민국 육군", org: "보안관제 · 체계관제", range: "2024.03 – 2025.09" },
  {
    title: "밑바닥부터 시작하는 딥러닝 스터디", org: "운영 · AID", range: "2023.03 – 2023.05", link: "https://github.com/Deep-Learning-from-Scratch-1",
    bullets: ["2주 1회 스터디 조직·운영, 챕터별 노트를 블로그에 기록 (→ Notes)"],
  },
  { title: "부산대학교 정보컴퓨터공학부", org: "학사 · 3학년", range: "2022.03 – 현재" },
];

export type Award = {
  year: string; name: string; org: string; result: string; soft?: boolean; link?: string;
};
export const awards: Award[] = [
  { year: "2026", name: "제3회 천문우주 AI 경진대회", org: "한국천문연구원 · KAIST", result: "5위", link: "https://kaist-kasiai.elice.io/" },
  { year: "2026", name: "DIVE 2026 부산시설공단×윌체어 트랙", org: "부산광역시 · 부산테크노파크", result: "3등 · 원장상", link: "https://www.dxchallenge.co.kr/dive-2026" },
  { year: "2026", name: "AI TOP 100 (CAMPUS)", org: "카카오임팩트 · 브라이언임팩트", result: "본선 100인", link: "https://www.etnews.com/20260406000039" },
  { year: "2026", name: "독서토론대회 『AI는 인간을 먹고 자란다』", org: "부산대학교 교양교육원", result: "대상 · 총장상", soft: true },
  { year: "2026", name: "AID Rummikub 에이전트 대회", org: "AID", result: "최우수상" },
  { year: "2024", name: "제23회 병영문학상 시 부문", org: "국방부", result: "입선", soft: true },
  { year: "2023", name: "제1회 건국대학교 해커톤", org: "건국대 SW중심대학사업단", result: "우수상", link: "https://github.com/Hackaton-Warriors/2023-Konkuk-Univ-HACKATON" },
  { year: "2023", name: "PNU CodeRace Beginner", org: "부산대 정보컴퓨터공학부", result: "은상", link: "https://www.acmicpc.net/contest/view/994" },
  { year: "2022", name: "PNU AI Landmark Classification · AI Art-Generation", org: "부산대 정보컴퓨터공학부", result: "금상 ×2" },
  { year: "2022", name: "2022 부산 코딩경진대회", org: "동명대 SW중심대학사업단", result: "동상" },
  { year: "2019", name: "제1회 인공지능기반 청소년캠프", org: "부산광역시교육청", result: "최우수상" },
];

export const certifications = ["PCCE Lv.3", "육상무선통신사", "무인동력비행장치 4종"];
export const beyond = ["📡 HAM 6K5EHL", "NOAA 위성 수신", "등산", "드론", "차량 수리"];
