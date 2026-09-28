/** Render the report with VS Code's actual math extension and embed all assets.
 * Usage:
 * node scripts/render_resilience_model_report.cjs VSCODE_SERVER_DIR MARKDOWN_IT_UMD
 * Optional third argument: original Markdown, for before/after diagnostics.
 * No project dependencies or VS Code settings are changed.
 */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const report = path.join(root, 'docs/resilience_model_group_meeting.md');
const output = path.join(root, 'docs/resilience_model_group_meeting.html');
const [server, markdownUmd, original] = process.argv.slice(2);
if (!server || !markdownUmd) throw new Error('Pass the VS Code server directory and Markdown-it UMD file.');

const markdownContext = { atob, btoa, TextEncoder, TextDecoder };
vm.runInNewContext(fs.readFileSync(markdownUmd, 'utf8'), markdownContext);
const MarkdownIt = markdownContext.markdownit;
const extensionModule = { exports: {} };
const vscodeStub = {
  workspace: {
    getConfiguration: () => ({ get: (_key, fallback) => fallback }),
    onDidChangeConfiguration: () => ({ dispose() {} }),
  },
  commands: { executeCommand() {} },
};
const extensionPath = path.join(server, 'extensions/markdown-math/dist/extension.js');
vm.runInNewContext(fs.readFileSync(extensionPath, 'utf8'), {
  module: extensionModule,
  exports: extensionModule.exports,
  require: name => {
    if (name === 'vscode') return vscodeStub;
    throw new Error(`Unexpected math-extension dependency: ${name}`);
  },
  console,
});
const extension = extensionModule.exports.activate({ subscriptions: [] });
const diagnostics = { renderer: extensionPath, reports: [] };
function render(file) {
  const markdown = extension.extendMarkdownIt(new MarkdownIt({ html: false }));
  const html = markdown.render(fs.readFileSync(file, 'utf8'));
  const errors = [...html.matchAll(/<(?:span|p)[^>]*class="[^"]*katex-error[\s\S]*?<\/(?:span|p)>/g)].map(m => m[0]);
  diagnostics.reports.push({
    file,
    rendered_formulas: (html.match(/class="katex"/g) || []).length,
    formula_errors: errors.length,
    internal_group_error: html.includes('Got group of unknown type'),
    errors,
  });
  return { html, errors };
}
if (original) render(original);
let { html, errors } = render(report);
if (errors.length) throw new Error(`The report still contains ${errors.length} formula rendering errors.`);

let imageCount = 0;
html = html.replace(/<img\b[^>]*\bsrc="([^"]+)"[^>]*>/g, (tag, src) => {
  const absolute = path.resolve(path.dirname(report), src);
  if (path.extname(absolute) !== '.png') throw new Error(`Unexpected image: ${src}`);
  const uri = `data:image/png;base64,${fs.readFileSync(absolute).toString('base64')}`;
  imageCount += 1;
  return tag.replace(src, uri);
});
if (imageCount !== 6) throw new Error(`Expected six figures, found ${imageCount}.`);

const dist = path.join(server, 'node_modules/katex/dist');
let mathCss = fs.readFileSync(path.join(dist, 'katex.min.css'), 'utf8');
// Browser-ready font faces: retain WOFF2 and embed them, avoiding external URLs.
mathCss = mathCss.replace(/src:url\(([^)]+\.woff2)\) format\("woff2"\)(?:,[^;]+)?;/g,
  (_match, font) => `src:url(data:font/woff2;base64,${fs.readFileSync(path.join(dist, font)).toString('base64')}) format("woff2");`);
if (/url\((?!data:)/.test(mathCss)) throw new Error('A font resource remains external.');

const style = `
* { box-sizing: border-box; }
html { color-scheme: light; scroll-behavior: smooth; }
body { margin: 0; background: #edf1f5; color: #263442; font: 16px/1.85 "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; }
main { max-width: 1220px; margin: 28px auto; padding: 48px 60px; background: white; border: 1px solid #dae1e8; border-radius: 12px; }
h1 { font-size: 28px; line-height: 1.5; color: #173955; }
h2 { margin-top: 2.5em; padding-bottom: .3em; border-bottom: 2px solid #dee7ef; color: #244b69; font-size: 23px; }
h3 { margin-top: 1.8em; font-size: 19px; color: #355a72; }
p { margin: 1em 0; }
blockquote { margin: 20px 0; padding: 2px 22px; border-left: 4px solid #7698b3; background: #f3f7fa; }
table { border-collapse: collapse; width: 100%; margin: 24px 0; font-size: 14px; }
th, td { border: 1px solid #d4dee6; padding: 9px 12px; }
th { background: #eef4f8; }
tr:nth-child(even) td { background: #fafcfd; }
img { display: block; max-width: 100%; height: auto; margin: 28px auto 16px; }
a { color: #226790; overflow-wrap: anywhere; }
code { padding: .12em .3em; background: #edf2f5; border-radius: 3px; overflow-wrap: anywhere; }
.katex-block { overflow-x: auto; overflow-y: hidden; padding: .5em 0; }
.katex { font-size: 1.08em; }
.katex-display { margin: .6em 0; }
.reading-note { color: #536879; font-size: 14px; padding: 12px 18px; background: #eef6f2; border-radius: 6px; }
@media(max-width: 800px) { main { margin: 0; padding: 24px 18px; border-radius: 0; } table { display: block; overflow-x: auto; } }
@media print { body { background: white; font-size: 10pt; } main { padding: 0; margin: 0; border: none; max-width: none; } h2,h3 { break-after: avoid; } img,tr,.katex-block { break-inside: avoid; } a { color: inherit; } .reading-note { display: none; } }
`;
const document = `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>全年容量规划与多级供电韧性模型</title><style>${mathCss}\n${style}</style></head>
<body><main><p class="reading-note">离线阅读版：${imageCount}张图片、${diagnostics.reports.at(-1).rendered_formulas}处公式及数学字体已嵌入本文件，无需联网或安装公式插件。模型内容与 Markdown 版一致。</p>
${html}</main></body></html>`;
fs.writeFileSync(output, document);
diagnostics.embedded_images = imageCount;
diagnostics.html_file = output;
diagnostics.html_bytes = Buffer.byteLength(document);
fs.writeFileSync(path.join(root, 'docs/assets/resilience_model_group_meeting/render_verification.json'), JSON.stringify(diagnostics, null, 2) + '\n');
console.log(JSON.stringify({ output, images: imageCount, formulas: diagnostics.reports.at(-1).rendered_formulas, errors: 0, bytes: diagnostics.html_bytes }, null, 2));
