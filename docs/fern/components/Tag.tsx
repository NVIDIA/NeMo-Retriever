/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Inline tag/chip rendered inside Card bodies.
 *
 * Mirrors sphinx-design `{bdg-*}` badges so legacy MyST landing pages keep
 * their visual cues after migration. Used by `convert_myst_to_fern.py` when
 * converting `{bdg-primary}\`text\`` to `<Tag variant="primary">text</Tag>`.
 *
 * Copy to your repo's `fern/components/` together with `BadgeLinks.tsx`.
 */
export type TagVariant =
  | "primary"
  | "secondary"
  | "success"
  | "warning"
  | "danger"
  | "info"
  | "light"
  | "dark";

const VARIANT_STYLES: Record<TagVariant, { bg: string; fg: string }> = {
  primary:   { bg: "#0d6efd", fg: "#ffffff" },
  secondary: { bg: "#6c757d", fg: "#ffffff" },
  success:   { bg: "#198754", fg: "#ffffff" },
  warning:   { bg: "#ffc107", fg: "#212529" },
  danger:    { bg: "#dc3545", fg: "#ffffff" },
  info:      { bg: "#0dcaf0", fg: "#212529" },
  light:     { bg: "#f8f9fa", fg: "#212529" },
  dark:      { bg: "#212529", fg: "#ffffff" },
};

export function Tag({
  variant = "secondary",
  children,
}: {
  variant?: TagVariant;
  children: React.ReactNode;
}) {
  const { bg, fg } = VARIANT_STYLES[variant] ?? VARIANT_STYLES.secondary;
  return (
    <span
      className={`fern-tag fern-tag--${variant}`}
      style={{
        display: "inline-block",
        padding: "2px 8px",
        marginRight: "6px",
        fontSize: "0.75rem",
        fontWeight: 500,
        lineHeight: 1.4,
        borderRadius: "0.375rem",
        background: bg,
        color: fg,
        verticalAlign: "baseline",
      }}
    >
      {children}
    </span>
  );
}
