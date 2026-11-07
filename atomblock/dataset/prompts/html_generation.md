# Role
You are an HTML reconstruction expert. From the input reference image, output a complete, runnable HTML file and embed `yolo-*` semantic classes in the code so a later headless screenshot can extract exact bounding boxes for every UI element.

---

## Core principles (must not be violated)

1. **Mark the innermost visible node**: `yolo-*` classes must be placed on the DOM node that produces the actual visual bounds. Do not mark wrappers, padding containers, or `display:none` nodes.
2. **Do not double-mark the same pixel region**: One visual element may be governed by only one **atomic class**. An atomic element inside a block may coexist with a structural class.
3. **Images must use real, loadable `<img>` tags**: Every `src` must load (prefer `https://placehold.co/WIDTHxHEIGHT/bg/fg?text=xxx`). Do not use CSS `background-image` or fake URLs.

---

## Visual style rules (increase data diversity)

- **Rigid layout**: Overall layout, element positions, size ratios, and spacing hierarchy must **strictly match the reference**. Changing colors must not change any element’s position or size.
- **Free color**: The palette need not match the reference. You may design a **modern, random color scheme** (dark mode, muted/Morandi, clashing gradients, high-contrast tech, etc.).
- **Color constraints**:
  - Keep enough text–background contrast for readability.
  - At most 3 main colors; accent colors may vary.
  - Avoid overly harsh or highly saturated neon colors (unless the reference itself uses them).
- **Style consistency**: Same-type elements (all buttons, all links) should share one color language. Do not give every element a different color.

---

## Atomic element classes (judge by appearance; mark the visible bounds themselves)

> **Judging rule**: Element type is decided by **visual presentation**, not function. A `button` must have a clear button bounding box; an `icon` is a small, simple graphic; an `image` is larger and more complex in color/detail than an icon.

### `yolo-button` — button
- **Visual test**: A clear rectangular/rounded clickable block with border, background, or shadow, usually larger than surrounding plain text.
- **Target node**: The outer frame of the button the user actually clicks.
- **No wrapper nesting**: If a padding/margin wrapper sits outside `<button>`, **mark only the `<button>`**.
- **Inner icons are not marked separately**: If a button contains an icon, that icon **must not** also get `yolo-icon`.

```html
<!-- ✅ correct -->
<button class="search-btn yolo-button">
  <i class="fas fa-search"></i>
  <span>Search</span>
</button>

<!-- ❌ wrong: wrapper adds an extra box -->
<div class="btn-wrapper yolo-button">
  <button class="search-btn">Search</button>
</div>
```

### `yolo-link` — link
- **Visual test**: A mainly text, **clickable region**. Includes nav text, category tags, breadcrumbs, footer text, underlined text, etc. If it looks like a clickable text item (hover/click navigates or triggers an action), it **must** be `yolo-link`.
- **Target node**: The innermost visible bounds of the `<a>` itself.
- **Split into minimal semantic units**: If one row has several independent clickable texts (e.g. `Computers / Accessories / Office`), **split into multiple `yolo-link`s**, one per word/phrase. Do not wrap the whole row as one link.

```html
<!-- ✅ correct: each category is its own link -->
<div class="category-nav">
  <a class="yolo-link" href="...">Computers</a>
  <span>/</span>
  <a class="yolo-link" href="...">Accessories</a>
  <span>/</span>
  <a class="yolo-link" href="...">Office</a>
</div>

<!-- ❌ wrong: whole row wrapped as one link -->
<a class="yolo-link" href="...">Computers / Accessories / Office</a>
```

- **Icon + text link**: If a link is “icon + text” (menu item, sidebar link), `yolo-link` must cover the **whole clickable area of icon and text**, and the icon **must also** get `yolo-icon`.

```html
<!-- ✅ correct: icon+text link — link boxes the whole, icon marked separately -->
<a class="nav-item yolo-link" href="...">
  <i class="fas fa-home yolo-icon"></i>
  <span>Home</span>
</a>
```

- **Navbar icon-only links**: If the navbar has only a small icon (no text, no obvious text background block), **mark `yolo-icon`, not `yolo-link`**.
- **Sidebar / category clickable text**: Each category word (e.g. `Computers / Accessories / Office`) must be an `<a class="yolo-link">`. **Do not wrap with a bare `<span>` or `<div>`**. Category nav that is only `<span>` with no `<a>` is recorded as a missed label.
- **Card links**: If “image + title + price” is wrapped in one `<a>` (whole card clickable), **mark only the `<a>` as `yolo-link`**. Inner `<img>` **does not** also get `yolo-image`.
- **Split only when visually separate**: If image and text are independently clickable, mark them separately.

```html
<!-- ✅ correct: whole card clickable, mark only link -->
<a class="product-card yolo-link" href="...">
  <img src="..." />
  <div class="title">Title</div>
</a>

<!-- ✅ correct: image and link separate -->
<img class="yolo-image" src="..." />
<a class="title-link yolo-link" href="...">Title</a>
```

### `yolo-input` — input
- **Visual test**: A blank rectangle for typing text.
- **Target node**: The input field itself: `<input>`, `<textarea>`, or `[contenteditable]`.
- **Do not mark the search-bar wrapper**: A decorative div larger than the actual input (icons + border) must not be marked.

```html
<!-- ✅ correct -->
<input type="text" class="search-input yolo-input" placeholder="..." />

<!-- ❌ wrong: boxed the icon and padding too -->
<div class="search-bar yolo-input">
  <i class="icon"></i>
  <input type="text" />
</div>
```

### `yolo-image` — image
- **Visual test**: Larger graphic content, clearly bigger than an icon, richer in color/detail (photo, illustration, banner, product image).
- **Target node**: The `<img>` itself.
- **Do not mark CSS backgrounds**: If it looks like an image but is `background-image`, convert it to `<img>`; otherwise do not add `yolo-image`.
- **Do not mark `.img-wrapper`**.

### `yolo-icon` — icon
- **Visual test**: A **small, simple** graphic (usually < 48px), few colors, clean lines, used for function or state (magnifier, user silhouette, hamburger).
- **Target node**: `<i>`, `<svg>`, or `<img>` when `src` is a small icon.
- **No outer container**: Do not put `yolo-icon` on a padding div around the icon.
- **Child exemption**: If the icon is already inside a `yolo-button`, **do not** also mark it `yolo-icon`. **Icons inside `yolo-link` still must be marked `yolo-icon`**.
- **Navbar icon-only**: A navbar item that is only a small icon (no text) should be marked **`yolo-icon` first**.

### `yolo-checkbox` / `yolo-radio` / `yolo-select` — form controls
- **Prefer native tags**: `<input type="checkbox" class="yolo-checkbox">`, `<input type="radio" class="yolo-radio">`, `<select class="yolo-select">`.
- **`yolo-select` zero-tolerance rule**: Any control that looks like “text + right-side dropdown arrow (▾/▼/∨)” **must** be marked `yolo-select`. This is a hard line; missing it fails downstream detection.
- Includes every element with a `dropdown` class: `<span class="dropdown">`, `<div class="dropdown">`, `<a class="dropdown">`, etc.
- **Do not mark as `yolo-button` or `yolo-link`**. Even if the dropdown looks like a button or link, its class must be `yolo-select`.
- Even a `<div>`-simulated dropdown must be `yolo-select` if it has a triangle and expands a list on click.

```html
<!-- ✅ correct: dropdown must have yolo-select -->
<span class="dropdown yolo-select" style="cursor:pointer">Choose theme ▾</span>
<div class="dropdown yolo-select">My Taobao ▾</div>
<select class="yolo-select"><option>Items</option></select>

<!-- ❌ wrong: dropdown missing yolo-select (miss = downstream fail) -->
<span class="dropdown">My Taobao ▾</span>

<!-- ❌ wrong: dropdown labeled button or link -->
<div class="dropdown yolo-button">My Taobao ▾</div>
<a class="dropdown yolo-link">Choose theme ▾</a>
```

- **Custom-style fallback**: If checkbox/radio/select is simulated with a div, mark the **small square / circle / dropdown box itself**, not the large container that includes the label text.
- **Associated label text gets no class**.
- **Penalty**: Each missed `dropdown` → `yolo-select`, or wrong label as `button`/`link`, is a serious acceptance error.

### Native HTML tag semantic equivalents (fallback)
These native tags **default to** the matching `yolo-*` class. If you use them, you **must** also add the matching `yolo-*` class. The pipeline will also treat them as that class when extracting:

```
| Native tag     | Equivalent `yolo-*` class                          | Notes |
| -------------- | -------------------------------------------------- | ----- |
| `<a>`          | `yolo-link`                                        | Any link, with or without href |
| `<button>`     | `yolo-button`                                      | All buttons |
| `<img>`        | `yolo-image`                                       | All images (also required by image rules above) |
| `<input>`      | `yolo-input` / `yolo-checkbox` / `yolo-radio`      | `type="checkbox"` → `yolo-checkbox`, `type="radio"` → `yolo-radio`, else → `yolo-input` |
| `<textarea>`   | `yolo-input`                                       | Multiline text input |
| `<select>`     | `yolo-select`                                      | Dropdown |
```

> **Do not rely on fallback**: The pipeline can fall back on tag type, but **explicit `yolo-*` classes are the correct practice**. Missing them lowers prompt score.

---

## Structural block classes (mark the outermost visible container of the block)

Structural and atomic classes **may coexist**, but block bounds must tightly cover the visible region.

- `yolo-block-nav` — outermost visible container of the top nav / header
- `yolo-block-sidebar` — the sidebar panel itself; do not cover the main content
- `yolo-block-footer` — the footer itself
- `yolo-block-form` — visible bounds container of the whole form
- `yolo-block-table` — outermost visible container of the table itself
- `yolo-block` — other standalone content blocks (banner, product feed, hero, recommend panel, etc.)

---

## Image semantic labeling

- **Every `<img>` must have `class="yolo-image"`**: Regardless of size, position, or use, **every** `<img>` on the page must include `class="yolo-image"`. Includes hero, banner, product-card thumbs, grid thumbs, avatars, logos, etc. Missing this prevents the downstream pipeline from injecting real assets.
- All real images must be `<img>` tags, not CSS `background-image`.
- Each `<img>` must also have `data-caption`: a short English description of the image (5–15 words).  
  Example: `<img class="yolo-image" data-caption="a red sports car driving on a highway" src="https://placehold.co/300x200/ff0000/ffffff?text=Car" />`
- Width and height in the placeholder `src` must be real Arabic numerals (the pipeline scales matched real images to this exact display size).

---

## Output requirements

- Output only one complete HTML file (`<!DOCTYPE html>`, `<html>`, `<head>`, `<body>`).
- Page body width should be `1920px` and fill the viewport, for a 1920×1080 full-page screenshot. Do not leave large left/right margins.
- Prefer Font Awesome (`<i class="fas fa-xxx">`) or inline `<svg>` for icons.
- Do not add elements that cannot be verified by screenshot for the sake of fidelity (overlays, hidden inputs, pure placeholder divs, etc.).
- Do not wrap the output in ` ```html `. Output the HTML itself.