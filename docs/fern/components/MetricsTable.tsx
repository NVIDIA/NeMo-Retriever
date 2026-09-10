/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * MetricsTable - Styled comparison table for benchmark results.
 *
 * Optional: highlights best values per column (bold).
 * NOTE: Fern's custom component pipeline uses the automatic JSX runtime.
 * Do NOT import React -- the `react` module is not resolvable in Fern's build.
 *
 * Usage in MDX:
 *   import { MetricsTable } from "@/components/MetricsTable";
 *
 *   <MetricsTable
 *     headers={["Variant", "Validation Loss", "Score"]}
 *     rows={[
 *       ["Baseline", "1.309", "36.99"],
 *       ["Ours", "1.256", "44.31"],
 *     ]}
 *     lowerIsBetter={[1]}
 *   />
 */

export interface MetricsTableProps {
  headers: string[];
  rows: (string | number)[][];
  /** Column indices where lower is better (for highlighting) */
  lowerIsBetter?: number[];
  /** Column indices where higher is better (default for non-lowerIsBetter) */
  higherIsBetter?: number[];
}

function findBestIndices(
  rows: (string | number)[][],
  colIndex: number,
  lowerIsBetter: boolean
): Set<number> {
  const values = rows.map((r) => {
    const v = r[colIndex];
    if (typeof v === "number") return v;
    const parsed = parseFloat(String(v));
    return isNaN(parsed) ? (lowerIsBetter ? Infinity : -Infinity) : parsed;
  });
  const best = lowerIsBetter ? Math.min(...values) : Math.max(...values);
  const bestIndices = new Set<number>();
  values.forEach((v, i) => {
    if (v === best) bestIndices.add(i);
  });
  return bestIndices;
}

export const MetricsTable = ({
  headers,
  rows,
  lowerIsBetter = [],
  higherIsBetter = [],
}: MetricsTableProps) => {
  const lowerSet = new Set(lowerIsBetter);
  const bestByCol: Record<number, Set<number>> = {};

  for (let c = 0; c < headers.length; c++) {
    if (lowerSet.has(c)) {
      bestByCol[c] = findBestIndices(rows, c, true);
    } else if (higherIsBetter.includes(c)) {
      bestByCol[c] = findBestIndices(rows, c, false);
    } else {
      const numLike = rows.every((r) => {
        const v = r[c];
        return typeof v === "number" || !isNaN(parseFloat(String(v)));
      });
      if (numLike) {
        bestByCol[c] = findBestIndices(rows, c, false);
      }
    }
  }

  return (
    <div className="metrics-table-wrapper">
      <table className="metrics-table">
        <thead>
          <tr>
            {headers.map((h, i) => (
              <th key={i}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIdx) => (
            <tr key={rowIdx} className={rowIdx % 2 === 1 ? "metrics-table__row--alt" : ""}>
              {row.map((cell, colIdx) => {
                const isBest = bestByCol[colIdx]?.has(rowIdx);
                return (
                  <td key={colIdx} className={isBest ? "metrics-table__cell--best" : ""}>
                    {cell}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
