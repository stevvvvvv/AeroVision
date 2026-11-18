import { useRef, useState } from "react";
import type { Detection } from "../api";

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
  defect_type: string;
}

interface Props {
  src: string;
  width: number;
  height: number;
  detections: Detection[];
  selected: number | null;
  onSelect: (id: number | null) => void;
  /** état de revue par détection : true = confirmé, false = rejeté */
  feedback?: Record<number, boolean | undefined>;
  addedBoxes?: Box[];
  drawMode?: boolean;
  onDraw?: (b: { x: number; y: number; w: number; h: number }) => void;
  showBoxes?: boolean;
}

/** Réticule : quatre coins plutôt qu'un rectangle plein, pour laisser voir le défaut. */
function bracketPath(x: number, y: number, w: number, h: number, len: number) {
  const l = Math.min(len, w / 2, h / 2);
  return [
    `M${x},${y + l} V${y} H${x + l}`,
    `M${x + w - l},${y} H${x + w} V${y + l}`,
    `M${x + w},${y + h - l} V${y + h} H${x + w - l}`,
    `M${x + l},${y + h} H${x} V${y + h - l}`,
  ].join(" ");
}

/** Image + zones en SVG (coordonnées = pixels de l'image, quel que soit le zoom). */
export default function ImageViewer({
  src, width, height, detections, selected, onSelect, feedback = {},
  addedBoxes = [], drawMode = false, onDraw, showBoxes = true,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const unit = Math.max(width, height) / 300;          // 1 « unité » ≈ 1/300 de l'image
  const stroke = Math.max(1.5, unit * 1.6);
  const tag = Math.max(11, unit * 9);

  const toImage = (e: React.MouseEvent) => {
    const r = svgRef.current!.getBoundingClientRect();
    return {
      x: Math.round(((e.clientX - r.left) / r.width) * width),
      y: Math.round(((e.clientY - r.top) / r.height) * height),
    };
  };

  return (
    <div className={`viewer ${drawMode ? "drawing" : ""}`}
         style={{ maxWidth: `min(100%, calc(72vh * ${width / height}))` }}>
      <img src={src} alt="Pièce inspectée" draggable={false} />
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        onMouseDown={(e) => {
          if (!drawMode) return;
          const p = toImage(e);
          setDrag({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
        }}
        onMouseMove={(e) => {
          if (!drag) return;
          const p = toImage(e);
          setDrag({ ...drag, x1: p.x, y1: p.y });
        }}
        onMouseUp={() => {
          if (drag && onDraw) {
            const b = {
              x: Math.min(drag.x0, drag.x1), y: Math.min(drag.y0, drag.y1),
              w: Math.abs(drag.x1 - drag.x0), h: Math.abs(drag.y1 - drag.y0),
            };
            if (b.w > 4 && b.h > 4) onDraw(b);
          }
          setDrag(null);
        }}
        onClick={(e) => !drawMode && e.target === svgRef.current && onSelect(null)}
      >
        {showBoxes &&
          detections.map((d) => {
            const rejected = feedback[d.id] === false;
            const confirmed = feedback[d.id] === true;
            const sel = selected === d.id;
            const color = rejected ? "#9aa6b1" : "#ff3b67";
            const pad = stroke * 2;
            const x = d.x - pad, y = d.y - pad, w = d.w + 2 * pad, h = d.h + 2 * pad;
            return (
              <g key={d.id} onClick={() => !drawMode && onSelect(d.id)} className="det">
                <rect x={x} y={y} width={w} height={h} fill={sel ? "rgba(255,59,103,0.10)" : "transparent"} />
                {rejected ? (
                  <rect x={x} y={y} width={w} height={h} fill="none" stroke={color} strokeWidth={stroke}
                        strokeDasharray={`${stroke * 3} ${stroke * 2}`} />
                ) : (
                  <path d={bracketPath(x, y, w, h, Math.max(unit * 10, Math.min(w, h) * 0.28))} fill="none"
                        stroke={color} strokeWidth={confirmed || sel ? stroke * 1.9 : stroke * 1.3}
                        strokeLinecap="square" />
                )}
                <g transform={`translate(${x}, ${Math.max(0, y - tag * 1.35)})`}>
                  <rect width={tag * 1.35} height={tag * 1.3} rx={tag * 0.18} fill={color} />
                  <text x={tag * 0.675} y={tag * 0.98} fontSize={tag} fontWeight={600} textAnchor="middle" fill="#fff"
                        fontFamily="Barlow Semi Condensed, Barlow, sans-serif">
                    {d.rank}
                  </text>
                </g>
              </g>
            );
          })}
        {addedBoxes.map((b, i) => (
          <rect key={`a${i}`} x={b.x} y={b.y} width={b.w} height={b.h} fill="rgba(80,150,255,0.12)"
                stroke="#5aa0ff" strokeWidth={stroke * 1.3} />
        ))}
        {drag && (
          <rect x={Math.min(drag.x0, drag.x1)} y={Math.min(drag.y0, drag.y1)}
                width={Math.abs(drag.x1 - drag.x0)} height={Math.abs(drag.y1 - drag.y0)}
                fill="none" stroke="#5aa0ff" strokeWidth={stroke} strokeDasharray={`${stroke * 4} ${stroke * 3}`} />
        )}
      </svg>
    </div>
  );
}
