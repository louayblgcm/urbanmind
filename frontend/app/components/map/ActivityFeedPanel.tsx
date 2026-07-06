"use client";

import {
  motion,
  AnimatePresence
} from "framer-motion";

import {
  AlertTriangle,
  ChevronDown,
  Clock3,
  MapPin,
  Siren,
  Car,
  ShieldAlert,
  Construction,
  X
} from "lucide-react";

import {
  useMemo,
  useState
} from "react";

type FeedEvent = {
  primary_type?: string;
  request_type?: string;
  service_name?: string;
  timestamp?: string;
  date?: string;
  created_at?: string;
  created_date?: string;
  description?: string;
  status?: string;
  details?: string;
  block?: string;
  address?: string;
  case_number?: string;
  arrest?: boolean;
  domestic?: boolean;
  feed_type?: "crime" | "311";
};

type SelectedArea = {
  raw_activity_feed?: {
    crimes?: FeedEvent[];
    requests_311?: FeedEvent[];
  };
};

// =====================================================
// HELPERS
// =====================================================

function getRelativeTime(
  timestamp: string
) {

  if (!timestamp) {
    return "Unknown";
  }

  const now =
    new Date().getTime();

  const then =
    new Date(timestamp).getTime();

  const diff =
    Math.max(
      0,
      Math.floor(
        (now - then) / 1000 / 60
      )
    );

  if (diff < 1) {
    return "Now";
  }

  if (diff < 60) {
    return `${diff}m ago`;
  }

  const hours =
    Math.floor(diff / 60);

  if (hours < 24) {
    return `${hours}h ago`;
  }

  const days =
    Math.floor(hours / 24);

  return `${days}d ago`;
}

// =====================================================
// EVENT STYLE
// =====================================================

function resolveEventStyle(
  event: FeedEvent
) {

  const type =
    (
      event.primary_type ||
      event.request_type ||
      ""
    ).toLowerCase();

  // ==========================================
  // VIOLENT
  // ==========================================

  if (
    type.includes("assault") ||
    type.includes("battery") ||
    type.includes("weapon") ||
    type.includes("homicide")
  ) {

    return {

      icon:
        <ShieldAlert size={13} />,

      color:
        "text-red-300",

      accent:
        "bg-red-400",

      glow:
        "hover:bg-red-400/[0.04]"
    };
  }

  // ==========================================
  // THEFT
  // ==========================================

  if (
    type.includes("theft") ||
    type.includes("burglary") ||
    type.includes("robbery") ||
    type.includes("vehicle")
  ) {

    return {

      icon:
        <Car size={13} />,

      color:
        "text-orange-300",

      accent:
        "bg-orange-400",

      glow:
        "hover:bg-orange-400/[0.04]"
    };
  }

  // ==========================================
  // 311
  // ==========================================

  if (
    type.includes("graffiti") ||
    type.includes("light") ||
    type.includes("street") ||
    type.includes("garbage") ||
    type.includes("rodent") ||
    type.includes("tree") ||
    type.includes("pothole")
  ) {

    return {

      icon:
        <Construction size={13} />,

      color:
        "text-yellow-300",

      accent:
        "bg-yellow-400",

      glow:
        "hover:bg-yellow-400/[0.04]"
    };
  }

  // ==========================================
  // DEFAULT
  // ==========================================

  return {

    icon:
      <AlertTriangle size={13} />,

    color:
      "text-cyan-300",

    accent:
      "bg-cyan-400",

    glow:
      "hover:bg-cyan-400/[0.04]"
  };
}

type Props = {
  selectedArea: SelectedArea | null;
  onClose: () => void;
};

// =====================================================
// COMPONENT
// =====================================================

export default function ActivityFeedPanel({
  selectedArea,
  onClose
}: Props) {

  const [expandedId, setExpandedId] =
    useState<string | null>(null);

  const [activeTab, setActiveTab] =
    useState("911");

  // =====================================================
  // FEEDS
  // =====================================================

  const crimes = useMemo(() => {

    return (

      selectedArea
        ?.raw_activity_feed
        ?.crimes || []

    )

      .map((crime: FeedEvent) => ({

        ...crime,

        feed_type:
          "crime",

        timestamp:
          crime.timestamp ||
          crime.date
      }))

      .sort(

        (a: FeedEvent, b: FeedEvent) =>

          new Date(
            b.timestamp || 0
          ).getTime()

          -

          new Date(
            a.timestamp || 0
          ).getTime()
      )

      .slice(0, 30);

  }, [selectedArea]);

  // =====================================================

  const requests311 = useMemo(() => {

    return (

      selectedArea
        ?.raw_activity_feed
        ?.requests_311 || []

    )

      .map((req: FeedEvent) => ({

        ...req,

        feed_type:
          "311",

        timestamp:
          req.created_at ||
          req.created_date ||
          req.timestamp
      }))

      .sort(

        (a: FeedEvent, b: FeedEvent) =>

          new Date(
            b.timestamp || 0
          ).getTime()

          -

          new Date(
            a.timestamp || 0
          ).getTime()
      )

      .slice(0, 30);

  }, [selectedArea]);

  // =====================================================

  const feed =

    activeTab === "911"

    ? crimes

    : requests311;

  // =====================================================

  if (!selectedArea) {

    return null;
  }

  // =====================================================
  // UI
  // =====================================================

  return (

    <motion.div

      initial={{
        opacity: 0,
        x: -25
      }}

      animate={{
        opacity: 1,
        x: 0
      }}

      exit={{
        opacity: 0,
        x: -25
      }}

      transition={{
        duration: 0.22
      }}

      className="
        fixed
        left-2
        right-2
        bottom-2

        z-[3000]

        h-[min(70dvh,680px)]
        max-h-[70dvh]

        sm:absolute
        sm:left-4
        sm:right-auto
        sm:top-[92px]
        sm:bottom-4
        sm:w-[315px]
        sm:h-full
        sm:max-h-[calc(100dvh-108px)]

        rounded-[30px]

        border
        border-white/[0.08]

        bg-[#0A0F14]/72

        backdrop-blur-[45px]
        backdrop-saturate-[180%]

        overflow-hidden

        text-white

        shadow-[0_12px_50px_rgba(0,0,0,0.34)]

        before:absolute
        before:inset-0
        before:bg-gradient-to-b
        before:from-white/[0.05]
        before:to-transparent
        before:pointer-events-none
      "
    >

      <div className="
        absolute
        left-1/2
        top-2.5
        z-20
        h-1.5
        w-12
        -translate-x-1/2
        rounded-full
        bg-white/16
        sm:hidden
      " />

      {/* HEADER */}

      <div className="
        relative

        px-5
        pt-7
        pb-4

        border-b
        border-white/[0.06]

        sm:pt-4
      ">

        {/* TOP */}

        <div className="
          flex
          items-center
          justify-between
        ">

          {/* LEFT */}

          <div className="
            flex
            items-center
            gap-3
          ">

            <div className="
              relative

              w-10
              h-10

              rounded-[14px]

              bg-white/[0.06]

              border
              border-white/[0.08]

              flex
              items-center
              justify-center

              text-cyan-300
            ">

              <Siren size={18} />

              {/* LIVE DOT */}

              <div className="
                absolute
                top-1.5
                right-1.5

                w-2
                h-2

                rounded-full

                bg-emerald-400

                animate-pulse
              " />

            </div>

            <div>

              <p className="
                text-[10px]

                uppercase

                tracking-[0.22em]

                text-white/35
              ">

                Live Area Feed

              </p>

              <h2 className="
                mt-1

                text-[20px]

                font-semibold

                tracking-tight
              ">

                Activity Feed

              </h2>

            </div>

          </div>

          {/* CLOSE */}

          <button

            onClick={onClose}

            className="
              w-9
              h-9

              rounded-[12px]

              bg-white/[0.05]

              border
              border-white/[0.06]

              flex
              items-center
              justify-center

              text-white/55

              hover:bg-white/[0.08]
              hover:text-white

              transition-all
            "
          >

            <X size={16} />

          </button>

        </div>

        {/* TABS */}

        <div className="
          mt-4

          flex

          rounded-[16px]

          bg-black/[0.16]

          p-1
        ">

          {["911", "311"].map((tab) => (

            <button

              key={tab}

              onClick={() =>
                setActiveTab(tab)
              }

              className={`
                flex-1

                py-2.5

                rounded-[12px]

                text-[12px]

                transition-all

                ${
                  activeTab === tab

                  ? `
                    bg-white/[0.9]

                    text-black

                    shadow-lg
                  `

                  : `
                    text-white/45
                  `
                }
              `}
            >

              {

                tab === "911"

                ? "911 Incidents"

                : "311 Services"
              }

            </button>
          ))}

        </div>

      </div>

      {/* FEED */}

      <div className="
        relative

        h-[calc(100%-145px)]

        overflow-y-auto
        no-scrollbar

        px-2
        py-2
        pb-[calc(env(safe-area-inset-bottom)+12px)]
      ">

        {feed.map((
          event: FeedEvent,
          index: number
        ) => {

          const style =
            resolveEventStyle(
              event
            );

          const id =
            `${event.feed_type}-${index}`;

          const expanded =
            expandedId === id;

          return (

            <motion.div

              key={id}

              initial={{
                opacity: 0,
                y: 10
              }}

              animate={{
                opacity: 1,
                y: 0
              }}

              transition={{
                delay:
                  index * 0.012
              }}

              className={`
                relative

                px-4
                py-3.5

                border-b
                border-white/[0.05]

                transition-all

                ${style.glow}
              `}
            >

              {/* LEFT ACCENT */}

              <div className={`
                absolute

                left-0
                top-3
                bottom-3

                w-[2px]

                rounded-full

                ${style.accent}
              `} />

              {/* CONTENT */}

              <div className="
                flex
                items-start
                gap-3
              ">

                {/* ICON */}

                <div className={`
                  mt-0.5

                  ${style.color}
                `}>

                  {style.icon}

                </div>

                {/* MAIN */}

                <div className="
                  flex-1
                  min-w-0
                ">

                  {/* HEADER */}

                  <div className="
                    flex
                    items-center
                    justify-between
                    gap-3
                  ">

                    <h3 className={`
                      text-[11px]

                      font-semibold

                      uppercase

                      tracking-[0.08em]

                      ${style.color}
                    `}>

                      {

                        event.primary_type ||

                        event.request_type ||

                        event.service_name ||

                        "Urban Event"
                      }

                    </h3>

                    <div className="
                      flex
                      items-center
                      gap-1

                      shrink-0

                      text-[10px]

                      text-white/28
                    ">

                      <Clock3 size={10} />

                      {

                        getRelativeTime(
                          event.timestamp
                        )
                      }

                    </div>

                  </div>

                  {/* DESCRIPTION */}

                  <p className="
                    mt-1.5

                    text-[12px]

                    leading-[1.6]

                    text-white/72
                  ">

                    {

                      event.description ||

                      event.status ||

                      event.details ||

                      event.request_type ||

                      event.block ||

                      event.address ||

                      "Operational urban event."
                    }

                  </p>

                  {/* LOCATION */}

                  {(

                    event.block ||

                    event.address

                  ) && (

                    <div className="
                      mt-2

                      flex
                      items-center
                      gap-1.5

                      text-[10px]

                      text-white/30
                    ">

                      <MapPin size={10} />

                      {

                        event.block ||

                        event.address
                      }

                    </div>
                  )}

                  {/* DETAILS BUTTON */}

                  <button

                    onClick={() =>

                      setExpandedId(

                        expanded

                        ? null

                        : id
                      )
                    }

                    className="
                      mt-2.5

                      flex
                      items-center
                      gap-1

                      text-[10px]

                      text-cyan-300/75

                      hover:text-cyan-200

                      transition-all
                    "
                  >

                    Details

                    <ChevronDown

                      size={11}

                      className={`
                        transition-transform

                        ${
                          expanded

                          ? "rotate-180"

                          : ""
                        }
                      `}
                    />

                  </button>

                  {/* EXPANDED */}

                  <AnimatePresence>

                    {expanded && (

                      <motion.div

                        initial={{
                          opacity: 0,
                          height: 0
                        }}

                        animate={{
                          opacity: 1,
                          height: "auto"
                        }}

                        exit={{
                          opacity: 0,
                          height: 0
                        }}

                        className="
                          mt-3

                          rounded-[14px]

                          bg-black/[0.18]

                          border
                          border-white/[0.05]

                          overflow-hidden
                        "
                      >

                        <div className="
                          p-3

                          space-y-2.5
                        ">

                          {

                            event.case_number && (

                              <div className="
                                flex
                                justify-between

                                text-[10px]
                              ">

                                <span className="
                                  text-white/35
                                ">
                                  Case
                                </span>

                                <span className="
                                  text-white/72
                                ">
                                  {
                                    event.case_number
                                  }
                                </span>

                              </div>
                            )
                          }

                          {

                            event.status && (

                              <div className="
                                flex
                                justify-between

                                text-[10px]
                              ">

                                <span className="
                                  text-white/35
                                ">
                                  Status
                                </span>

                                <span className="
                                  text-white/72
                                ">
                                  {event.status}
                                </span>

                              </div>
                            )
                          }

                          {

                            event.arrest !== undefined && (

                              <div className="
                                flex
                                justify-between

                                text-[10px]
                              ">

                                <span className="
                                  text-white/35
                                ">
                                  Arrest
                                </span>

                                <span className="
                                  text-white/72
                                ">
                                  {
                                    event.arrest

                                    ? "Yes"

                                    : "No"
                                  }
                                </span>

                              </div>
                            )
                          }

                          {

                            event.domestic !== undefined && (

                              <div className="
                                flex
                                justify-between

                                text-[10px]
                              ">

                                <span className="
                                  text-white/35
                                ">
                                  Domestic
                                </span>

                                <span className="
                                  text-white/72
                                ">
                                  {
                                    event.domestic

                                    ? "Yes"

                                    : "No"
                                  }
                                </span>

                              </div>
                            )
                          }

                        </div>

                      </motion.div>
                    )}

                  </AnimatePresence>

                </div>

              </div>

            </motion.div>
          );
        })}

      </div>

    </motion.div>
  );
}
