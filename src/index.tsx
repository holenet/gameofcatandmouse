import { render } from "preact";

import "./style.css";
import { useEffect, useRef } from "preact/hooks";
import { useComputed, useSignal } from "@preact/signals";
import classnames from "classnames";
import CatSvg from "./catSvg";
import IntroScene from "./introScene";

const DIFFICULTIES = ["Easy", "Hard", "Very Hard", "Impossible"] as const;
type Difficulty = (typeof DIFFICULTIES)[number];

const normalizeRotation = (angle: number) => {
  while (angle <= -Math.PI) angle += Math.PI * 2;
  while (angle >= Math.PI) angle -= Math.PI * 2;
  return angle;
};

function App() {
  const svgSize = useSignal(0);
  const scale = 1.5;

  const difficulty = useSignal<Difficulty>("Easy");

  const mouseSpeed = useSignal(0.5);
  const speedFactor = useComputed(() => {
    switch (difficulty.value) {
      case "Easy":
        return 4.0;
      case "Hard":
        return 4.2;
      case "Very Hard":
        return 4.4;
      case "Impossible":
        return 4.5;
      default:
        return 4.0;
    }
  });
  const catSpeed = useComputed(() => mouseSpeed.value * speedFactor.value);
  const catRotationSpeed = useComputed(() => catSpeed.value);

  const targetCoord = useSignal({ x: 0, y: -0.000001 });
  const mouseCoord = useSignal({ x: 0, y: -0.000001 });
  const mouseFacing = useSignal(Math.PI);
  const catRotation = useSignal((Math.random() * 2 - 1) * Math.PI);
  const catCoord = useComputed(() => ({ x: Math.cos(catRotation.value), y: Math.sin(catRotation.value) }));
  const catFacing = useSignal(0);
  const betweenAngle = useComputed(() => {
    const mouseRotation = Math.atan2(mouseCoord.value.y, mouseCoord.value.x);
    return normalizeRotation(mouseRotation - catRotation.value);
  });

  const svgRef = useRef<SVGSVGElement>();
  const svgContainerRef = useRef<HTMLDivElement>();

  useEffect(() => {
    const svgContainer = svgContainerRef.current;
    if (svgContainer) {
      const observer = new ResizeObserver((entries) => {
        for (let entry of entries) {
          svgSize.value = Math.min(entry.contentRect.width, entry.contentRect.height);
        }
      });
      observer.observe(svgContainer);
      return () => observer.unobserve(svgContainer);
    }
  }, [svgContainerRef]);
  useEffect(() => {
    const svgElement = svgRef.current;
    if (svgElement) {
      const mouseMove = (ev: MouseEvent) => {
        const size = svgElement.getBoundingClientRect().width;
        targetCoord.value = {
          x: (ev.offsetX / size - 0.5) * scale * 2,
          y: (ev.offsetY / size - 0.5) * scale * 2,
        };
      };
      svgElement.addEventListener("mousemove", mouseMove);

      const touchMove = (ev: TouchEvent) => {
        const svgRect = svgElement.getBoundingClientRect();
        const size = svgRect.width;
        targetCoord.value = {
          x: ((ev.touches[0].clientX - svgRect.x) / size - 0.5) * scale * 2,
          y: ((ev.touches[0].clientY - svgRect.y) / size - 0.5) * scale * 2,
        };
      };

      svgElement.addEventListener("touchmove", touchMove);
      svgElement.addEventListener("touchstart", touchMove);
      return () => {
        svgElement.removeEventListener("mousemove", mouseMove);
        svgElement.removeEventListener("touchstart", touchMove);
        svgElement.removeEventListener("touchmove", touchMove);
      };
    }
  }, [svgRef]);

  const isFirst = useSignal(true);
  const isPlaying = useSignal(false);
  const isMouseWin = useSignal(false);
  const firstMessage = useComputed(() =>
    isFirst.value ? "Click to Start" : isMouseWin.value ? "YOU WIN!" : "YOU DIED!"
  );
  const secondMessage = useComputed(() => (isFirst.value ? null : "Click to Restart"));
  const lastTickTime = useSignal(0);
  const lastStartTime = useSignal(0);
  const lastTimeRecord = useSignal(0);
  useEffect(() => {
    if (isPlaying.value) {
      lastTimeRecord.value = lastTickTime.value - lastStartTime.value;
    }
  }, [lastTickTime.value]);
  const bestTimeRecords = useSignal<{ [key in Difficulty]: number | null }>({
    Easy: null,
    Hard: null,
    "Very Hard": null,
    Impossible: null,
  });
  const mouseWinOffset = useComputed(() => {
    if (isPlaying.value || !isMouseWin.value) return { x: 0, y: 0 };
    const timeElapsed = lastTickTime.value - lastStartTime.value - lastTimeRecord.value;
    const speed = timeElapsed * 3;
    let dx = mouseCoord.value.x - catCoord.value.x;
    let dy = mouseCoord.value.y - catCoord.value.y;
    const d = Math.hypot(dx, dy);
    dx *= speed / d;
    dy *= speed / d;
    return { x: dx, y: dy };
  });
  const catAnimatedStep = useComputed(() => (isPlaying.value ? ~~((lastTickTime.value * 1000) / 500) % 2 : 0));
  const mouseAnimatedStep = useComputed(() => (isPlaying.value ? ~~((lastTickTime.value * 1000) / 750) % 2 : 0));
  const tick = (time: number) => {
    const deltaTime = Math.min(0.033333333, time - lastTickTime.value);
    lastTickTime.value = time;

    if (!isPlaying.value) return;

    let dx = targetCoord.value.x - mouseCoord.value.x;
    let dy = targetCoord.value.y - mouseCoord.value.y;
    const d = Math.hypot(dx, dy);
    const prevCoord = mouseCoord.value;
    if (d <= 0.000001) {
      mouseCoord.value = targetCoord.value;
    } else {
      const limitSpeed = mouseSpeed.value * deltaTime;
      if (d > limitSpeed) {
        dx *= limitSpeed / d;
        dy *= limitSpeed / d;
      }
      mouseCoord.value = {
        x: mouseCoord.value.x + dx,
        y: mouseCoord.value.y + dy,
      };
      mouseFacing.value = Math.atan2(mouseCoord.value.y - prevCoord.y, mouseCoord.value.x - prevCoord.x);
    }

    if (Math.abs(betweenAngle.value) > 0.000001) {
      const catRotationLimitSpeed = catRotationSpeed.value * deltaTime;
      let d = Math.sign(betweenAngle.value) * catRotationLimitSpeed;
      if (Math.abs(betweenAngle.value) < Math.abs(d)) {
        d = betweenAngle.value;
      }
      catRotation.value = normalizeRotation(catRotation.value + d);
      catFacing.value = Math.abs(d) < 0.005 ? 0 : Math.sign(d);
    } else {
      catFacing.value = 0;
    }

    const catMouseDistance = Math.hypot(mouseCoord.value.x - catCoord.value.x, mouseCoord.value.y - catCoord.value.y);
    if (catMouseDistance <= 0.01) {
      isMouseWin.value = false;
      isPlaying.value = false;
    } else if (Math.hypot(mouseCoord.value.x, mouseCoord.value.y) >= 1) {
      const mouseWin = catMouseDistance >= 0.1;
      isMouseWin.value = mouseWin;
      isPlaying.value = false;
      if (mouseWin) {
        const prevRadius = Math.hypot(prevCoord.x, prevCoord.y);
        const currRadius = Math.hypot(mouseCoord.value.x, mouseCoord.value.y);
        const recordTime = lastTimeRecord.value + (deltaTime * (1 - prevRadius)) / (currRadius - prevRadius);
        lastTimeRecord.value = recordTime;
        bestTimeRecords.value = {
          ...bestTimeRecords.value,
          [difficulty.value]: Math.min(bestTimeRecords.value[difficulty.value] ?? Number.MAX_SAFE_INTEGER, recordTime),
        };
        mouseFacing.value = Math.atan2(mouseCoord.value.y - catCoord.value.y, mouseCoord.value.x - catCoord.value.x);
      }
    }
  };

  useEffect(() => {
    let lastAnimFunc = 0;
    const anim = (time: number) => {
      tick(time / 1000);
      lastAnimFunc = requestAnimationFrame(anim);
    };
    anim(0);
    return () => cancelAnimationFrame(lastAnimFunc);
  }, []);

  const reset = () => {
    isFirst.value = false;
    isPlaying.value = true;
    lastStartTime.value = lastTickTime.value;
    isMouseWin.value = false;
    mouseCoord.value = { x: 0, y: 0 };
    catRotation.value = (Math.random() * 2 - 1) * Math.PI;
  };
  const tryReset = () => {
    if (!isPlaying.value) reset();
  };

  useEffect(() => {
    const svgElement = svgRef.current;
    if (svgElement) {
      svgElement.addEventListener("mousedown", reset);
      return () => svgElement.removeEventListener("mousedown", reset);
    }
  }, [svgRef]);
  useEffect(() => {
    const svgElement = svgRef.current;
    if (svgElement) {
      svgElement.addEventListener("touchstart", tryReset);
      return () => svgElement.removeEventListener("touchstart", tryReset);
    }
  }, [svgRef]);

  const showHint = useSignal(false);
  useEffect(() => {
    const onKeyPress = (ev: KeyboardEvent) => {
      if (ev.code === "KeyH") showHint.value = !showHint.value;
      else if (ev.code === "KeyD" && !isPlaying.value)
        difficulty.value = DIFFICULTIES[(DIFFICULTIES.indexOf(difficulty.value) + 1) % DIFFICULTIES.length];
    };
    window.addEventListener("keydown", onKeyPress);
    return () => window.removeEventListener("keydown", onKeyPress);
  }, []);

  const showIntro = useSignal(true);

  return (
    <div class="flex flex-col w-full h-full justify-center items-stretch">
      {showIntro.value && (
        <div class="absolute w-full h-full flex justify-center items-center z-10 bg-black">
          <IntroScene onEnded={() => (showIntro.value = false)} />
        </div>
      )}
      <div class="text-6xl font-bold mt-12" style={{ fontFamily: "fantasy", textWrap: "balance" }}>
        Game of Cat and Mouse
      </div>
      <div class="flex flex-col gap-1">
        <div
          className={classnames("font-bold", "text-3xl", "mt-4", "mb-8", {
            "text-green-500": isMouseWin.value,
            "text-red-500": !isMouseWin.value,
          })}
          style={{ opacity: isPlaying.value ? 0 : 100 }}
        ></div>
        <div
          className={classnames("flex", "flex-wrap", "gap-2", "items-center", "justify-center", {
            "gap-y-8": Object.values(bestTimeRecords.value).some((x) => x > 0),
          })}
        >
          {DIFFICULTIES.map((d) => (
            <div class="relative">
              <button
                className={classnames(
                  "outline",
                  "outline-2",
                  "hover:bg-neutral-100",
                  "px-2",
                  "py-0.5",
                  "min-w-28",
                  "rounded-sm",
                  "font-semibold",
                  {
                    "outline-green-300 text-green-500": d === "Easy",
                    "outline-orange-300 text-orange-500": d === "Hard",
                    "outline-red-300 text-red-500": d === "Very Hard",
                    "outline-purple-300 text-purple-500": d === "Impossible",
                    "outline-4 shadow-md": difficulty.value === d,
                  }
                )}
                onClick={(e) => {
                  e.stopPropagation();
                  difficulty.value = d;
                }}
              >
                {d}
              </button>
              {bestTimeRecords.value[d] !== null && (
                <div class="absolute top-full mt-1 w-full text-center">{bestTimeRecords.value[d].toFixed(3)} s</div>
              )}
            </div>
          ))}
        </div>
        <div className={classnames("mt-6", { "opacity-0": isFirst.value })}>{lastTimeRecord.value.toFixed(3)} s</div>
      </div>
      <div class="grow shrink basis-0 relative">
        <div class="w-full h-full relative overflow-hidden" ref={svgContainerRef}>
          <svg
            class="mx-auto"
            width={svgSize}
            height={svgSize}
            viewBox={`${-scale} ${-scale} ${scale * 2} ${scale * 2}`}
            ref={svgRef}
          >
            <defs>
              <radialGradient id="RadialGradient1">
                <stop offset="0%" stop-color="#ACF" />
                <stop offset="50%" stop-color="#CDF" />
                <stop offset="100%" stop-color="#DEF" />
              </radialGradient>
            </defs>
            <circle cx={0} cy={0} r={1} stroke={"#CCC"} stroke-width={0.005} fill="url(#RadialGradient1)" />
            {showHint.value && (
              <>
                <circle cx={0} cy={0} r={0.25} stroke={"#CCC5"} stroke-width={0.005} fill={"#FDF7"} />
                <circle
                  cx={0}
                  cy={0}
                  r={1 - Math.PI / speedFactor.value}
                  stroke={"#CCC4"}
                  stroke-width={0.005}
                  fill={"#FED7"}
                />
                <line
                  x1={0}
                  y1={0}
                  x2={mouseCoord.value.x}
                  y2={mouseCoord.value.y}
                  stroke="#000"
                  stroke-width={0.001}
                />
                <line x1={0} y1={0} x2={catCoord.value.x} y2={catCoord.value.y} stroke="#A00" stroke-width={0.001} />
              </>
            )}
            <g
              transform={`rotate(${(catRotation.value * 180) / Math.PI + 90}) translate(0 -1) scale(${
                catFacing.value > 0 ? 1 : -1
              } 1)`}
              fill="#333"
              style="filter: drop-shadow(0 0 0.01px rgba(0, 0, 0, 0.4))"
            >
              <CatSvg
                isPlaying={isPlaying}
                facing={catFacing}
                animatedStep={catAnimatedStep}
                fill="transparent"
                stroke="#e7a7b1"
                strokeWidth="40"
              />
              <CatSvg
                isPlaying={isPlaying}
                facing={catFacing}
                animatedStep={catAnimatedStep}
                fill="#333"
                stroke="transparent"
                strokeWidth="0"
              />
            </g>
            <g transform={`translate(${mouseWinOffset.value.x} ${mouseWinOffset.value.y})`}>
              <g
                transform={`translate(${mouseCoord.value.x} ${mouseCoord.value.y}) rotate(${
                  (mouseFacing.value * 180) / Math.PI + 90
                }) scale(${mouseAnimatedStep.value === 0 ? 1 : -1} 1)`}
                fill="#333"
                style="filter: drop-shadow(0 0 0.01px rgba(0, 0, 0, 0.4))"
              >
                <g transform="scale(0.0002) translate(-175 -350)">
                  <path
                    d="M411.97,600.96c-.3-9.37-2.55-18.6-7.03-26.85-9.11-16.76-26.19-27.92-44.09-34.5-47.78-17.56-99.73-6.99-149.26-10.8-5.19-.4-10.25-.89-15.11-1.59-4.89-.65-9.57-1.54-14.1-2.51-1.56-.35-3.1-.73-4.62-1.12,86.17-2.5,155.25-73.13,155.25-159.9,0-63.87-37.44-119-91.57-144.64-1.22-1.89-1.82-3.6-1.74-5.3.3-6.13,7.71-10.55,9.45-11.63,15.16-9.44,52.27-56.77,63.92-100.86,2.59-9.79,5.34-24.81,4.09-44.38,2.85-1.71,11.15-7.14,14.39-17.59.88-2.84,1.36-5.86,1.36-8.99,0-16.73-13.56-30.3-30.3-30.3s-30.3,13.56-30.3,30.3c0,3.66.65,7.17,1.84,10.42-21.8-.41-42.58,4.15-63.04,11.11-9.05,3.08-17.83,6.68-26.44,10.77-5.22-16.75-20.85-28.92-39.32-28.92-8.39,0-16.2,2.52-22.71,6.83,2.52,3.2,5,6.97,7.14,11.39,6.51,13.43,6.79,26.4,5.82,35.21-.28-16.71-6.61-31.96-16.89-43.63-12.43-14.11-30.62-23.01-50.9-23.01C30.36,20.46,0,50.82,0,88.27s30.36,67.81,67.81,67.81c7.39,0,14.51-1.19,21.17-3.38-.14.49-.29.99-.47,1.53-7.18,21.4-2.66,41.37,10.33,59.31,1.82,2.52,2.95,4.66,3.28,6.71-52.78,26.14-89.08,80.55-89.08,143.44,0,53.78,26.54,101.36,67.24,130.37.59,1.19,1.21,2.39,1.88,3.6,3.56,6.49,8.16,13.17,13.9,19.64,5.73,6.48,12.59,12.76,20.65,18.42,31.96,22.45,71.88,27.03,109.93,26.09,35.06-.86,70.22-5.6,105.07-1.74,17.14,1.9,35.17,6.52,46.73,19.31,7.84,8.68,11.87,20.67,10.86,32.32-1.72,20-17.2,36.24-34.19,46.92-35.19,22.14-80.19,29.23-121.04,31.91-12.58.83-25.01,1.08-37.12.7-12.25-.44-24.22-.96-35.71-2.13-45.2-4.6-89.84-23.47-118.67-59.92-1.27-1.6-2.46-3.25-3.7-4.87-4.23-5.76-7.75-11.33-10.71-16.48-2.93-5.17-5.28-9.95-7.23-14.15-1.88-4.23-3.41-7.87-4.52-10.86-.58-1.49-1.06-2.82-1.46-3.97-.41-1.15-.75-2.12-1.01-2.9-.52-1.57-.79-2.38-.79-2.38,0,0,.23.82.68,2.41.22.8.51,1.78.87,2.95.35,1.17.76,2.52,1.28,4.04.97,3.06,2.33,6.77,4.02,11.11,1.76,4.31,3.89,9.23,6.61,14.59,15.11,29.85,39.36,55.07,69.2,70.39,59.46,30.54,138.18,29.27,202.44,18.23,20.95-3.6,41.87-8.76,60.77-18.48,18.9-9.72,35.79-24.35,45.29-43.37,5.3-10.61,8.04-22.66,7.67-34.48ZM200.76,82.11c7.61,0,13.78,6.17,13.78,13.78s-6.17,13.78-13.78,13.78-13.78-6.17-13.78-13.78,6.17-13.78,13.78-13.78Z"
                    fill="transparent"
                    stroke="#DDD"
                    stroke-width="50"
                  />
                  <path
                    d="M411.97,600.96c-.3-9.37-2.55-18.6-7.03-26.85-9.11-16.76-26.19-27.92-44.09-34.5-47.78-17.56-99.73-6.99-149.26-10.8-5.19-.4-10.25-.89-15.11-1.59-4.89-.65-9.57-1.54-14.1-2.51-1.56-.35-3.1-.73-4.62-1.12,86.17-2.5,155.25-73.13,155.25-159.9,0-63.87-37.44-119-91.57-144.64-1.22-1.89-1.82-3.6-1.74-5.3.3-6.13,7.71-10.55,9.45-11.63,15.16-9.44,52.27-56.77,63.92-100.86,2.59-9.79,5.34-24.81,4.09-44.38,2.85-1.71,11.15-7.14,14.39-17.59.88-2.84,1.36-5.86,1.36-8.99,0-16.73-13.56-30.3-30.3-30.3s-30.3,13.56-30.3,30.3c0,3.66.65,7.17,1.84,10.42-21.8-.41-42.58,4.15-63.04,11.11-9.05,3.08-17.83,6.68-26.44,10.77-5.22-16.75-20.85-28.92-39.32-28.92-8.39,0-16.2,2.52-22.71,6.83,2.52,3.2,5,6.97,7.14,11.39,6.51,13.43,6.79,26.4,5.82,35.21-.28-16.71-6.61-31.96-16.89-43.63-12.43-14.11-30.62-23.01-50.9-23.01C30.36,20.46,0,50.82,0,88.27s30.36,67.81,67.81,67.81c7.39,0,14.51-1.19,21.17-3.38-.14.49-.29.99-.47,1.53-7.18,21.4-2.66,41.37,10.33,59.31,1.82,2.52,2.95,4.66,3.28,6.71-52.78,26.14-89.08,80.55-89.08,143.44,0,53.78,26.54,101.36,67.24,130.37.59,1.19,1.21,2.39,1.88,3.6,3.56,6.49,8.16,13.17,13.9,19.64,5.73,6.48,12.59,12.76,20.65,18.42,31.96,22.45,71.88,27.03,109.93,26.09,35.06-.86,70.22-5.6,105.07-1.74,17.14,1.9,35.17,6.52,46.73,19.31,7.84,8.68,11.87,20.67,10.86,32.32-1.72,20-17.2,36.24-34.19,46.92-35.19,22.14-80.19,29.23-121.04,31.91-12.58.83-25.01,1.08-37.12.7-12.25-.44-24.22-.96-35.71-2.13-45.2-4.6-89.84-23.47-118.67-59.92-1.27-1.6-2.46-3.25-3.7-4.87-4.23-5.76-7.75-11.33-10.71-16.48-2.93-5.17-5.28-9.95-7.23-14.15-1.88-4.23-3.41-7.87-4.52-10.86-.58-1.49-1.06-2.82-1.46-3.97-.41-1.15-.75-2.12-1.01-2.9-.52-1.57-.79-2.38-.79-2.38,0,0,.23.82.68,2.41.22.8.51,1.78.87,2.95.35,1.17.76,2.52,1.28,4.04.97,3.06,2.33,6.77,4.02,11.11,1.76,4.31,3.89,9.23,6.61,14.59,15.11,29.85,39.36,55.07,69.2,70.39,59.46,30.54,138.18,29.27,202.44,18.23,20.95-3.6,41.87-8.76,60.77-18.48,18.9-9.72,35.79-24.35,45.29-43.37,5.3-10.61,8.04-22.66,7.67-34.48ZM200.76,82.11c7.61,0,13.78,6.17,13.78,13.78s-6.17,13.78-13.78,13.78-13.78-6.17-13.78-13.78,6.17-13.78,13.78-13.78Z"
                    fill="current"
                  />
                </g>
              </g>
            </g>
            {!isPlaying.value && (
              <>
                <text
                  font-size="0.2"
                  text-anchor="middle"
                  alignment-baseline="baseline"
                  y="0.07"
                  font-weight="bold"
                  stroke="white"
                  stroke-width={0.02}
                  style="
                    user-select: none;
                    pointer-events: none;
                    filter: drop-shadow(0 0 0.02px rgba(0, 0, 0, 0.2));
                  "
                >
                  {firstMessage}
                </text>
                <text
                  font-size="0.2"
                  text-anchor="middle"
                  alignment-baseline="baseline"
                  y="0.07"
                  font-weight="bold"
                  fill={isFirst.value ? "#333" : isMouseWin.value ? "#3D6" : "#F33"}
                  stroke="transparent"
                  style="
                    user-select: none;
                    pointer-events: none;
                    filter: drop-shadow(0 0 0.01px rgb(255, 255, 255));
                  "
                >
                  {firstMessage}
                </text>
              </>
            )}
          </svg>
        </div>
      </div>
      <div
        class="absolute right-1 bottom-1 select-none pointer-events-none font-bold opacity-25"
        style="filter: invert(100%)"
      >
        Special Thanks to Lucas Kang
      </div>
    </div>
  );
}

render(<App />, document.getElementById("app"));
