// 마크다운 표 접근성 보강.
// - 표 바로 앞 문단이 "표: …" / "Table: …" 이면 화면에서 숨긴 <caption>으로 옮긴다
// - thead 의 th 에 scope="col"
// - 가로 스크롤 래퍼로 감싸고, 캡션이 있으면 키보드로 스크롤할 수 있는 region으로 만든다
const CAPTION = /^\s*(?:표|Table)\s*:\s*/;

const el = (tagName, properties, children) => ({ type: "element", tagName, properties, children });
const find = (node, test) => test(node) ? node : (node.children ?? []).reduce((hit, c) => hit ?? find(c, test), undefined);
const isKatex = (node) => node.properties?.className?.includes("katex");
const plainText = (node) => node.type === "text" ? node.value : isKatex(node) ? "" : (node.children ?? []).map(plainText).join("");
const mathText = (node) => node.type === "text" ? node.value : node.tagName === "annotation" ? "" : (node.children ?? []).map(mathText).join("");

function scopeHeaders(node) {
  for (const child of node.children ?? []) {
    if (child.type !== "element") continue;
    if (child.tagName !== "th") { scopeHeaders(child); continue; }
    child.properties.scope = "col";
    // 수식만 든 머리글은 브라우저가 이름을 계산하지 못해 MathML 텍스트로 이름을 준다
    const math = !plainText(child).trim() && find(child, (n) => n.tagName === "math");
    if (math) child.properties.ariaLabel = mathText(math).trim();
  }
}

export default function rehypeTableA11y() {
  return (tree) => {
    let count = 0;
    const walk = (parent) => {
      const kids = parent.children ?? [];
      for (let i = 0; i < kids.length; i++) {
        const node = kids[i];
        if (node.type !== "element") continue;
        if (node.tagName !== "table") { walk(node); continue; }

        for (const part of node.children) if (part.type === "element" && part.tagName === "thead") scopeHeaders(part);

        let j = i - 1;
        while (j >= 0 && kids[j].type === "text" && !kids[j].value.trim()) j--;
        const prev = kids[j];
        const wrap = el("div", { className: ["table-wrap"] }, [node]);
        const first = prev?.type === "element" && prev.tagName === "p" ? prev.children[0] : undefined;
        if (first?.type === "text" && CAPTION.test(first.value)) {
          first.value = first.value.replace(CAPTION, "");
          const id = `table-caption-${++count}`;
          node.children.unshift(el("caption", { id, className: ["sr-only"] }, prev.children));
          Object.assign(wrap.properties, { role: "region", tabIndex: 0, ariaLabelledBy: id });
          kids.splice(j, i - j);
          i = j;
        }
        kids[i] = wrap;
      }
    };
    walk(tree);
  };
}
