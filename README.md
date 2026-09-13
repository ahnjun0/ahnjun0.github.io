# ahnjun0.github.io

안준영 포트폴리오. Astro로 만들고 GitHub Pages에 배포합니다.

- 홈 콘텐츠(소개·경력·수상): `src/config.ts`
- 프로젝트: `src/content/projects/*.md` — `featured: 1~4`가 홈에 노출
- 노트: `src/content/notes/*.md`
- 스타일: `src/styles/global.css` (토큰 하나, 포인트 컬러 하나)

```bash
npm install
npm run dev      # http://localhost:4321
npm run build    # dist/
```

`main`에 push하면 `.github/workflows/deploy.yml`이 빌드·배포합니다.
