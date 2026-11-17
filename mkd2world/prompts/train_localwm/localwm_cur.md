# WM_Local

Given the current screenshot, generate the complete visible interface Markdown.

## Input

- `current_screenshot`: the current webpage screenshot.

## Output Format

Output Markdown directly:

[Environment: Web]
...

Rules:
- Markdown must start with `[Environment: Web]`.



## Markdown Grammar

Markdown uses a 0..999 coordinate grid relative to current_screenshot.
Each UI element uses center-box coordinates:

`[content](type)<cx,cy,w,h>`

Meaning:
- `cx,cy` are the element center point.
- `w,h` are the element width and height.
- All four numbers are integers on the `0..999` grid.

Valid line formats:
- `[text](text)<cx,cy,w,h>`
- `[text](link)<cx,cy,w,h>`
- `[text](button)<cx,cy,w,h>`
- `[visible value or placeholder](input)<cx,cy,w,h>`
- `[icon: semantic description](icon)<cx,cy,w,h>`
- `[img: semantic description](image)<cx,cy,w,h>`
- `[ ] [label](toggle)<cx,cy,w,h>`
- `[x] [label](toggle)<cx,cy,w,h>`
- `[>] [label](select)<cx,cy,w,h>`
- `[v] [label](select)<cx,cy,w,h>`
- `- [Block: description]<cx,cy,w,h>`
- `[Environment: Web]`

Element type definitions:
- `text`: visible non-interactive text.
- `link`: visible text or content that navigates when activated.
- `button`: An element with a visible border and a color that clearly contrasts with the webpage background. Its content is usually text, and it is generally larger than an icon.
- `input`: a visible editable field; use its current value, otherwise its visible placeholder.
- `icon`: A small graphical symbol whose background color usually matches the webpage background. It may optionally have a black outline.
- `image: A photograph or visual depicting natural scenes, people, or similar content, with boundaries clearly distinguishable from the webpage background.
- `toggle`: Any checkable element, such as a checkbox or radio button.
- `select`: A dropdown or expandable selection control.



### Toggle State

- `[ ]` indicates unchecked, off, or unselected.
- `[x]` indicates checked, on, or selected.

### Select State

- `[>]` indicates collapsed or closed.
- `[v]` indicates expanded or open.

### Block

Block description must be one of:
- `Navigation`
- `Sidebar`
- `Footer`
- `Form`
- `Table`
- `Section`

### Concrete Example


[Environment: Web]

- [Block: Navigation]<500,40,960,60>
  - [Home](link)<80,40,80,30>
  - [icon: menu](icon)<950,40,30,30>
- [Block: Sidebar]<110,350,180,520>
  - [ ] [Dark](toggle)<100,150,120,30>
  - [x] [Sync](toggle)<100,200,120,30>
  - [>] [More](select)<100,250,120,30>
- [Block: Form]<590,210,720,200>
  - [Query](input)<520,190,480,35>
  - [Go](button)<850,190,100,35>
- [Block: Section]<590,410,720,160>
  - [v] [Mode](select)<340,370,160,30>
  - [sample1](text)<340,415,160,30>
  - [sample2](text)<340,460,160,30>
- [Block: Table]<590,600,720,160>
  - [Row](text)<360,580,200,25>
- [Block: Footer]<500,960,960,60>
  - [img: blue mountain lake at sunset](image)<160,960,240,30>


## Rules

The top-to-bottom order of elements in the prediction should roughly match their order in the webpage viewport.
Prioritize correct semantics and element types, then infer reasonable coordinate positions.
