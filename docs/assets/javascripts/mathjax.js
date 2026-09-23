window.MathJax = {
  loader: {
    load: ["[tex]/boldsymbol"],
  },
  tex: {
    packages: { "[+]": ["boldsymbol"] },
    inlineMath: [
      ["\\(", "\\)"],
      ["$", "$"],
    ],
    displayMath: [["\\[", "\\]"]],
  },
  options: {
    processHtmlClass: "arithmatex",
    ignoreHtmlClass: ".*|",
  },
};

document$.subscribe(() => {
  MathJax.typesetPromise();
});
