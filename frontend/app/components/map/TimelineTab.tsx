"use client";

import { Clock3 } from "lucide-react";
import { useMemo, useState } from "react";

type TimelinePoint = {
  hour?: number;
  value?: number;
};

type MetricsShape = {
  crime_timeline?: {
    hourly_distribution?: TimelinePoint[];
  };
  forecast_timeline?: {
    hourly_distribution?: TimelinePoint[];
  };
  forecast?: {
    predicted_crime?: number;
    static_baseline_crime?: number;
  };
};

type Props = {
  metrics: MetricsShape;
};

const HOURS = Array.from({ length: 24 }, (_, hour) => hour);

function normalizeTimeline(series: TimelinePoint[] = []) {
  const byHour = new Map<number, number>();

  series.forEach((point) => {
    const hour = Number(point?.hour);
    const value = Number(point?.value || 0);

    if (Number.isFinite(hour)) {
      byHour.set(hour, value);
    }
  });

  return HOURS.map((hour) => ({
    hour,
    value: Math.max(Number(byHour.get(hour) || 0), 0),
  }));
}

function labelForValue(value: number) {
  if (value >= 80) {
    return "Extreme";
  }

  if (value >= 60) {
    return "High";
  }

  if (value >= 35) {
    return "Moderate";
  }

  if (value >= 15) {
    return "Low";
  }

  return "Quiet";
}

function labelForDelta(delta: number) {
  if (delta >= 8) {
    return "Clearly above typical";
  }

  if (delta >= 3) {
    return "Slightly above typical";
  }

  if (delta <= -8) {
    return "Clearly below typical";
  }

  if (delta <= -3) {
    return "Slightly below typical";
  }

  return "Close to typical";
}

export default function TimelineTab({ metrics }: Props) {
  const [timelineMode, setTimelineMode] =
    useState<"current" | "forecast">("current");

  const staticTimeline = useMemo(
    () =>
      normalizeTimeline(
        metrics?.crime_timeline?.hourly_distribution || []
      ),
    [metrics]
  );

  const forecastTimeline = useMemo(
    () =>
      normalizeTimeline(
        metrics?.forecast_timeline?.hourly_distribution || []
      ),
    [metrics]
  );

  const timelineData =
    timelineMode === "current"
      ? staticTimeline
      : forecastTimeline;

  const peakHour = timelineData.length
    ? timelineData.reduce((max, current) =>
        current.value > max.value ? current : max
      )
    : null;

  const peakDelta = useMemo(() => {
    if (!peakHour || timelineMode !== "forecast") {
      return 0;
    }

    const staticAtPeak =
      staticTimeline.find((point) => point.hour === peakHour.hour)?.value || 0;

    return peakHour.value - staticAtPeak;
  }, [peakHour, staticTimeline, timelineMode]);

  const totals = useMemo(() => {
    const predicted = Number(
      metrics?.forecast?.predicted_crime || 0
    );
    const baseline = Number(
      metrics?.forecast?.static_baseline_crime || 0
    );
    const delta = predicted - baseline;

    return {
      predicted,
      baseline,
      delta,
    };
  }, [metrics]);

  const timelineSemantic = useMemo(() => {
    if (!peakHour) {
      return "Limited crime activity in the trained static pattern.";
    }

    const hour = Number(peakHour.hour);
    const period =
      hour <= 4
        ? "late-night hours"
        : hour <= 10
          ? "morning hours"
          : hour <= 16
            ? "daytime hours"
            : hour <= 21
              ? "evening hours"
              : "late-evening hours";

    if (timelineMode === "forecast") {
      const direction =
        peakDelta > 3
          ? "above"
          : peakDelta < -3
            ? "below"
            : "near";

      return `Forecast crime pressure peaks during ${period}, ${direction} the trained static pattern for that hour.`;
    }

    return `The trained static crime pattern reaches its highest level during ${period}.`;
  }, [peakDelta, peakHour, timelineMode]);

  const forecastChangeLabel =
    labelForDelta(totals.delta);

  return (
    <div
      className="
        rounded-[26px]
        border
        border-white/10
        bg-white/[0.03]
        p-4
        sm:p-5
      "
    >
      <div
        className="
          flex
          items-start
          justify-between
        "
      >
        <div>
          <div
            className="
              flex
              items-center
              gap-2
            "
          >
            <Clock3
              size={18}
              className="text-cyan-300"
            />

            <h2
              className="
                text-[15px]
                font-medium
              "
            >
              Crime Pressure Timeline
            </h2>
          </div>

          <p
            className="
              mt-2
              max-w-[320px]
              text-[12px]
              leading-relaxed
              text-white/50
            "
          >
            {timelineMode === "current"
              ? "Hourly crime pattern learned by the static training model."
              : "AI-estimated crime pressure for the next 24 hours, with static context shown softly in the background."}
          </p>
        </div>
      </div>

      <div
        className="
          mt-5
          flex
          w-full
          sm:inline-flex
          rounded-2xl
          border
          border-white/10
          bg-white/[0.04]
          p-1
        "
      >
        <button
          onClick={() => setTimelineMode("current")}
          className={`
            rounded-xl
            flex-1
            px-3
            py-2.5
            text-[12px]
            font-medium
            transition-all
            ${
              timelineMode === "current"
                ? "bg-cyan-400 text-black"
                : "text-white/55"
            }
          `}
        >
          Static Baseline
        </button>

        <button
          onClick={() => setTimelineMode("forecast")}
          className={`
            rounded-xl
            flex-1
            px-3
            py-2.5
            text-[12px]
            font-medium
            transition-all
            ${
              timelineMode === "forecast"
                ? "bg-orange-400 text-black"
                : "text-white/55"
            }
          `}
        >
          Next 24h Forecast
        </button>
      </div>

      {timelineMode === "forecast" && (
        <div
          className="
            mt-5
            grid
            gap-3
            sm:grid-cols-3
          "
        >
          <div
            className="
              rounded-2xl
              border
              border-orange-300/15
              bg-gradient-to-br
              from-orange-400/10
              to-pink-500/5
              p-4
              shadow-[0_0_28px_rgba(251,146,60,0.08)]
            "
          >
            <p className="text-[11px] uppercase tracking-[0.14em] text-orange-100/55">
              AI forecast (24h)
            </p>

            <p className="mt-2 text-[24px] font-semibold text-white">
              {totals.predicted.toFixed(2)}
            </p>

            <p className="mt-1 text-[11px] text-orange-100/55">
              Predicted reported-crime pressure
            </p>
          </div>

          <div
            className="
              rounded-2xl
              border
              border-white/8
              bg-white/[0.03]
              p-4
            "
          >
            <p className="text-[11px] uppercase tracking-[0.14em] text-white/40">
              Static baseline (24h)
            </p>

            <p className="mt-2 text-[22px] font-medium text-white/88">
              {totals.baseline.toFixed(2)}
            </p>

            <p className="mt-1 text-[11px] text-white/35">
              Background reference
            </p>
          </div>

          <div
            className="
              rounded-2xl
              border
              border-white/8
              bg-white/[0.03]
              p-4
            "
          >
            <p className="text-[11px] uppercase tracking-[0.14em] text-white/40">
              Net change
            </p>

            <p
              className={`
                mt-2
                text-[20px]
                font-medium
                ${
                  totals.delta > 0.01
                    ? "text-orange-300"
                    : totals.delta < -0.01
                      ? "text-cyan-300"
                      : "text-white"
                }
              `}
            >
              {totals.delta > 0 ? "+" : ""}
              {totals.delta.toFixed(2)}
            </p>

            <p className="mt-1 text-[11px] text-white/35">
              {forecastChangeLabel}
            </p>
          </div>
        </div>
      )}

      <div
        className="
          mt-5
          rounded-2xl
          border
          border-white/8
          bg-white/[0.03]
          p-4
        "
      >
        <div
          className="
            flex
            items-center
            justify-between
          "
        >
          <p className="text-[12px] text-white/45">
            {timelineMode === "forecast"
              ? "Peak Forecast Crime"
              : "Static Crime Peak"}
          </p>

          <p className="text-[12px] font-medium text-white">
            {peakHour
              ? `${String(peakHour.hour).padStart(2, "0")}:00`
              : "--:--"}
          </p>
        </div>

        <p
          className="
            mt-3
            text-[13px]
            leading-relaxed
            text-white/70
          "
        >
          {timelineSemantic}
        </p>

        <p className="mt-3 text-[11px] text-white/30">
          {timelineMode === "forecast"
            ? "The brighter bars show the next-24-hour forecast, while the softer bars behind them show the usual static pattern."
            : "Leakage-safe hourly crime baseline learned during static training."}
        </p>
      </div>

      <div
        className="
          mt-6
          rounded-[24px]
          border
          border-white/6
          bg-black/10
          p-3
          sm:p-4
        "
      >
        <div
          className="
            mb-4
            flex
            flex-wrap
            items-center
            gap-4
            text-[10px]
            text-white/35
          "
        >
          {timelineMode === "forecast" ? (
            <>
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-white/35" />
                <span>Static baseline</span>
              </div>

              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-orange-300" />
                <span>AI forecast</span>
              </div>
            </>
          ) : (
            <div className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-cyan-300" />
              <span>Static baseline</span>
            </div>
          )}
        </div>

        <div
          className="
            relative
            flex
            h-[180px]
            sm:h-[170px]
            items-end
            gap-[4px]
            sm:gap-[5px]
          "
        >
          <div
            className="
              absolute
              left-0
              right-0
              bottom-1/2
              z-0
              border-t
              border-dashed
              border-white/20
            "
          >
            <span
              className="
                absolute
                -top-4
                right-0
                text-[9px]
                text-white/30
              "
            >
              Static avg
            </span>
          </div>

          {timelineData.map((point, index) => {
            const value =
              timelineMode === "current"
                ? Math.max(point.value, 4)
                : Math.max(forecastTimeline[index]?.value || 0, 4);
            const baselineValue = Math.max(
              staticTimeline[index]?.value || 0,
              4
            );
            const hour =
              timelineMode === "current"
                ? point.hour
                : forecastTimeline[index]?.hour ?? point.hour ?? index;
            const label = labelForValue(value);
            const delta = value - baselineValue;

            return (
              <div
                key={hour}
              className="
                z-10
                flex
                h-full
                flex-1
                items-end
                justify-center
              "
            >
              <div
                  title={
                    timelineMode === "forecast"
                      ? `${String(hour).padStart(2, "0")}:00 - Forecast ${value.toFixed(1)} vs static ${baselineValue.toFixed(1)}`
                      : `${String(hour).padStart(2, "0")}:00 - ${label} Crime Pressure`
                  }
                  className="
                    relative
                    flex
                    h-full
                    w-full
                    items-end
                    justify-center
                  "
                >
                  {timelineMode === "forecast" && (
                    <div
                      className="
                        absolute
                        bottom-0
                        w-full
                        rounded-full
                        border
                        border-white/8
                        bg-white/16
                        backdrop-blur-sm
                      "
                      style={{
                        height: `${baselineValue}%`,
                      }}
                    />
                  )}

                  <div
                    className={`
                      relative
                      rounded-full
                      transition-all
                      ${
                        timelineMode === "current"
                          ? `
                            w-full
                            bg-gradient-to-t
                            from-cyan-300
                            via-blue-400
                            to-indigo-500
                          `
                          : `
                            w-[72%]
                            bg-gradient-to-t
                            from-orange-300
                            via-red-400
                            to-pink-500
                            shadow-[0_0_22px_rgba(251,146,60,0.24)]
                          `
                      }
                    `}
                    style={{
                      height: `${value}%`,
                    }}
                  />

                  {timelineMode === "forecast" && Math.abs(delta) >= 6 && (
                    <span
                      className={`
                        absolute
                        -top-5
                        text-[9px]
                        font-medium
                        ${
                          delta > 0
                            ? "text-orange-200"
                            : "text-cyan-200"
                        }
                      `}
                    >
                      {delta > 0 ? "+" : ""}
                      {delta.toFixed(0)}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        <div
          className="
            mt-4
            flex
            justify-between
            text-[10px]
            text-white/30
          "
        >
          <span>12AM</span>
          <span>6AM</span>
          <span>12PM</span>
          <span>6PM</span>
          <span>11PM</span>
        </div>
      </div>
    </div>
  );
}
