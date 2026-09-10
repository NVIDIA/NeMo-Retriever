/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * CustomCard - Simple card with title, text, link, and optional sparkle.
 *
 * Alternative to Fern's built-in <Card> when you need custom styling
 * (e.g. devnotes/blog landing pages).
 * NOTE: Fern's custom component pipeline uses the automatic JSX runtime.
 *
 * Usage in MDX:
 *   import { CustomCard } from "@/components/CustomCard";
 *   <CustomCard title="Title" text="Description" link="/path" sparkle />
 */

export interface CustomCardProps {
  title: string;
  text: string;
  link: string;
  sparkle?: boolean;
}

export const CustomCard = ({ title, text, link, sparkle = false }: CustomCardProps) => {
  return (
    <a href={link} className="block p-6 rounded-lg border border-gray-200 hover:shadow-lg transition-shadow">
      <h2 className="text-xl font-semibold mb-2">
        {title} {sparkle && "✨"}
      </h2>
      <p className="text-gray-600">{text}</p>
    </a>
  );
};
