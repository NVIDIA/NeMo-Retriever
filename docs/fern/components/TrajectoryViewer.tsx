/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * TrajectoryViewer - Renders multi-turn research trajectories with tool calls.
 *
 * Displays search, open, find, and answer steps with color-coded styling.
 * Used for deep research / MCP tool-use dev notes.
 *
 * NOTE: Fern's custom component pipeline uses the automatic JSX runtime.
 * Do NOT import React -- the `react` module is not resolvable in Fern's build.
 *
 * Usage in MDX:
 *   import { TrajectoryViewer } from "@/components/TrajectoryViewer";
 *   import trajectory from "@/components/devnotes/<post-slug>/<example>";
 *
 *   <TrajectoryViewer {...trajectory} defaultOpen />
 */

export interface ToolCall {
  fn: "search" | "open" | "find" | "answer";
  arg?: string;
  body?: string;
  isGolden?: boolean;
}

export interface TrajectoryTurn {
  turnIndex: number;
  calls: ToolCall[];
}

export interface TrajectoryViewerProps {
  question: string;
  referenceAnswer?: string;
  goldenPassageHint?: string;
  turns: TrajectoryTurn[];
  summary?: string;
  defaultOpen?: boolean;
}

const TOOL_ICONS: Record<string, string> = {
  search: "🔍",
  open: "📄",
  find: "🔎",
  answer: "✓",
};

function ToolCallBlock({ call }: { call: ToolCall }) {
  const isAnswer = call.fn === "answer";
  const argDisplay = call.arg ?? "";
  const cn = `trajectory-viewer__call trajectory-viewer__call--${call.fn}`;
  const icon = TOOL_ICONS[call.fn] ?? "";

  if (isAnswer && call.body) {
    return (
      <div className={cn}>
        <span className="trajectory-viewer__fn">
          {icon && <span className="trajectory-viewer__icon">{icon}</span>}
          {call.fn}
        </span>
        <div
          className="trajectory-viewer__body"
          dangerouslySetInnerHTML={{ __html: call.body }}
        />
      </div>
    );
  }

  return (
    <div className={cn}>
      <span className="trajectory-viewer__fn">
        {icon && <span className="trajectory-viewer__icon">{icon}</span>}
        {call.fn}
      </span>
      <span className="trajectory-viewer__arg">
        {argDisplay}
        {call.isGolden && " ⭐"}
      </span>
    </div>
  );
}

export const TrajectoryViewer = ({
  question,
  referenceAnswer,
  goldenPassageHint,
  turns,
  summary,
  defaultOpen = false,
}: TrajectoryViewerProps) => {
  const content = (
    <div className="trajectory-viewer">
      <div className="trajectory-viewer__question">
        <strong>Q:</strong> {question}
      </div>
      {referenceAnswer && (
        <div className="trajectory-viewer__ref">
          <strong>Reference:</strong> {referenceAnswer}
        </div>
      )}
      {goldenPassageHint && (
        <div className="trajectory-viewer__hint">{goldenPassageHint}</div>
      )}
      <div className="trajectory-viewer__turns">
        {turns.map((turn) => (
          <div key={turn.turnIndex} className="trajectory-viewer__turn">
            <div className="trajectory-viewer__label">T{turn.turnIndex}</div>
            <div className="trajectory-viewer__body">
              <div
                className={`trajectory-viewer__group ${
                  turn.calls.length > 1 ? "trajectory-viewer__group--multi" : ""
                }`}
              >
                {turn.calls.map((call, i) => (
                  <ToolCallBlock key={i} call={call} />
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const totalCalls = turns.reduce((acc, t) => acc + t.calls.length, 0);

  if (summary) {
    return (
      <details className="trajectory-viewer__details" open={defaultOpen}>
        <summary className="trajectory-viewer__summary">
          <strong>{summary}</strong>
          <span className="trajectory-viewer__stats">
            {turns.length} turns · {totalCalls} calls
          </span>
        </summary>
        {content}
      </details>
    );
  }

  return content;
};
