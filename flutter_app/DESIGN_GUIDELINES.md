# Pixel Sorcery Design Guidelines

## Core Principles

1. **Clean & Minimal** - No visual clutter. Every element must serve a purpose.
2. **Custom Components Only** - NO Material UI. All components are custom-built.
3. **Subtle Animations** - Smooth, purposeful animations that enhance UX without being distracting.
4. **Dark Theme First** - Optimized for the dark theme with careful contrast ratios.

## Color Palette

### Primary Colors
- Background: `#0A0A0B` (near black)
- Surface: `#141416` (elevated surfaces)
- Surface Light: `#1C1C1F` (cards, inputs)
- Border: `#2A2A2D` (subtle borders)

### Accent Colors
- Primary: `#6366F1` (indigo - main accent)
- Primary Light: `#818CF8` (hover states)
- Success: `#10B981` (completion states)
- Warning: `#F59E0B` (caution states)
- Error: `#EF4444` (error states)

### Text Colors
- Primary Text: `#FAFAFA` (main content)
- Secondary Text: `#A1A1AA` (supporting text)
- Muted Text: `#71717A` (disabled, hints)

## Typography

### Font Family
- **Inter** - Clean, modern sans-serif optimized for screens

### Scale
- Display: 32px, Bold (700)
- Heading 1: 24px, SemiBold (600)
- Heading 2: 20px, SemiBold (600)
- Heading 3: 16px, Medium (500)
- Body: 14px, Regular (400)
- Caption: 12px, Regular (400)
- Tiny: 10px, Medium (500)

## Spacing

Use 4px base unit:
- xs: 4px
- sm: 8px
- md: 16px
- lg: 24px
- xl: 32px
- 2xl: 48px

## Border Radius

- None: 0px
- Small: 6px (buttons, chips)
- Medium: 12px (cards, inputs)
- Large: 16px (modals, sheets)
- Full: 9999px (circular elements)

## Shadows

Minimal shadow usage. When needed:
- Subtle: `0 1px 2px rgba(0,0,0,0.4)`
- Medium: `0 4px 12px rgba(0,0,0,0.5)`
- Heavy: `0 8px 24px rgba(0,0,0,0.6)`

## Animation Guidelines

### Durations
- Quick: 150ms (micro-interactions)
- Normal: 250ms (standard transitions)
- Slow: 400ms (page transitions, modals)

### Easing
- Default: `Curves.easeOutCubic`
- Bounce: `Curves.elasticOut` (success states only)
- Spring: Custom physics for drag interactions

## Component Patterns

### Buttons
- Primary: Filled with primary color, white text
- Secondary: Transparent with border, primary text
- Ghost: No border, primary text, hover background

### Cards
- Surface color background
- Subtle border (1px)
- Medium border radius
- No shadows by default

### Loading States
- Shimmer effect for content loading
- Pulsing animation for processing
- Progress indicators show actual progress when available

## Accessibility

- Minimum touch target: 44x44px
- Color contrast ratio: 4.5:1 minimum for text
- Focus indicators for keyboard navigation
- Support for system font scaling
