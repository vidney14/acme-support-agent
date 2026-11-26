// Minimal stroke icons (24px grid, currentColor) so the UI needs no icon library.
import type { SVGProps } from "react";

type P = SVGProps<SVGSVGElement>;
const base = {
  width: 18,
  height: 18,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

export const MicIcon = (p: P) => (
  <svg {...base} {...p}>
    <rect x="9" y="3" width="6" height="11" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
  </svg>
);
export const MicOffIcon = (p: P) => (
  <svg {...base} {...p}>
    <path d="M15 9.5V6a3 3 0 0 0-5.8-1M9 9v2a3 3 0 0 0 4.6 2.5M5 11a7 7 0 0 0 11.3 5.5M19 11a7 7 0 0 1-.6 2.8M12 18v3M3 3l18 18" />
  </svg>
);
export const PhoneOffIcon = (p: P) => (
  <svg {...base} {...p}>
    <path d="M4.5 15.5c-1-1-1-2.6.3-3.6A13 13 0 0 1 12 9.5a13 13 0 0 1 7.2 2.4c1.3 1 1.3 2.6.3 3.6l-1.3 1.3a1.5 1.5 0 0 1-1.9.2l-1.8-1.2a1.5 1.5 0 0 1-.6-1.5l.2-1.1a11 11 0 0 0-4.2 0l.2 1.1a1.5 1.5 0 0 1-.6 1.5l-1.8 1.2a1.5 1.5 0 0 1-1.9-.2z" />
  </svg>
);
export const SendIcon = (p: P) => (
  <svg {...base} {...p}>
    <path d="M5 12h13M13 6l6 6-6 6" />
  </svg>
);
export const PanelIcon = (p: P) => (
  <svg {...base} {...p}>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <path d="M15 4v16" />
  </svg>
);
export const BookIcon = (p: P) => (
  <svg {...base} {...p}>
    <path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2zM4 21V5M8 7h7" />
  </svg>
);
export const CloseIcon = (p: P) => (
  <svg {...base} {...p}>
    <path d="M6 6l12 12M18 6 6 18" />
  </svg>
);
export const CheckIcon = (p: P) => (
  <svg {...base} {...p}>
    <path d="m5 12.5 4.5 4.5L19 7.5" />
  </svg>
);
export const AlertIcon = (p: P) => (
  <svg {...base} {...p}>
    <path d="M12 4 2.5 20h19zM12 10v4.5M12 17.5v.01" />
  </svg>
);
export const RefreshIcon = (p: P) => (
  <svg {...base} {...p}>
    <path d="M20 11a8 8 0 0 0-14.6-4.5L4 8M4 4v4h4M4 13a8 8 0 0 0 14.6 4.5L20 16M20 20v-4h-4" />
  </svg>
);
export const HandoffIcon = (p: P) => (
  <svg {...base} {...p}>
    <circle cx="9" cy="8" r="3.5" />
    <path d="M3 20c.6-3.5 3-5.5 6-5.5 1.5 0 2.8.4 3.8 1.2M16 14h6M19 11l3 3-3 3" />
  </svg>
);
export const WaveIcon = (p: P) => (
  <svg {...base} {...p}>
    <path d="M3 12h2M7 8v8M11 5v14M15 9v6M19 7v10M21 12h0" />
  </svg>
);
export const SparkIcon = (p: P) => (
  <svg {...base} {...p}>
    <path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M18 6l-2.5 2.5M8.5 15.5 6 18" />
  </svg>
);
export const ReceiptIcon = (p: P) => (
  <svg {...base} {...p}>
    <path d="M6 3h12v18l-3-2-3 2-3-2-3 2zM9 8h6M9 12h6" />
  </svg>
);
export const BugIcon = (p: P) => (
  <svg {...base} {...p}>
    <rect x="7" y="8" width="10" height="12" rx="5" />
    <path d="M9 8a3 3 0 0 1 6 0M3 13h4M17 13h4M4 7l3 2M20 7l-3 2M4 19l3-2M20 19l-3-2" />
  </svg>
);
export const CloudIcon = (p: P) => (
  <svg {...base} {...p}>
    <path d="M7 18a4.5 4.5 0 0 1-.5-9A6 6 0 0 1 18 9.5a4 4 0 0 1-.5 8.5z" />
  </svg>
);
export const ShieldIcon = (p: P) => (
  <svg {...base} {...p}>
    <path d="M12 3 4.5 6v5.5c0 4.5 3.2 8 7.5 9.5 4.3-1.5 7.5-5 7.5-9.5V6zM9 12l2 2 4-4" />
  </svg>
);
export const BoltIcon = (p: P) => (
  <svg {...base} {...p}>
    <path d="M13 3 5 13.5h6L10 21l8-10.5h-6z" />
  </svg>
);
export const PulseIcon = (p: P) => (
  <svg {...base} {...p}>
    <path d="M3 12h4l2.5-6 5 12 2.5-6h4" />
  </svg>
);
