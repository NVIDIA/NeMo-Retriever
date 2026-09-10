/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Badge links for GitHub, License, PyPI, etc.
 * Uses a custom wrapper to avoid Fern's external-link icon stacking under badges.
 *
 * `badges` is required — there is intentionally no default. A previous
 * version shipped placeholder URLs that could land in production for
 * sites that rendered the component without props. See README-BadgeLinks.md.
 */
export type BadgeItem = {
  href: string;
  src: string;
  alt: string;
};

export interface BadgeLinksProps {
  badges: BadgeItem[];
}

export function BadgeLinks({ badges }: BadgeLinksProps) {
  return (
    <div
      className="badge-links"
      style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}
    >
      {badges.map((b) => (
        <a key={b.href} href={b.href} target="_blank" rel="noreferrer">
          <img src={b.src} alt={b.alt} />
        </a>
      ))}
    </div>
  );
}
