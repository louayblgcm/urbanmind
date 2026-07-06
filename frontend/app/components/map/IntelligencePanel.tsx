"use client";

import { useState } from "react";

import { motion } from "framer-motion";

import {
  Shield,
  X
} from "lucide-react";

import OverviewTab from "./OverviewTab";
import SignalsTab from "./SignalTab";
import TimelineTab from "./TimelineTab";
import AITab, {
  type ChatMessage
} from "./AITab";
import { apiUrl } from "@/app/lib/api";

type MetricValue = {
  value?: number | string;
  percentage?: number | string;
};

type SelectedArea = {
  loading?: boolean;
  context_id?: string;
  overview?: string;
  context?: {
    location_name?: string;
  };
  metrics?: {
    safety_score?: MetricValue;
    theft_risk?: MetricValue;
    violence_risk?: MetricValue;
    nightlife_activity?: MetricValue;
    urban_vitality?: MetricValue;
    workplace_activity?: MetricValue;
    transit_access?: MetricValue;
    urban_personality?: {
      value?: string;
    };
  };
};

type Props = {
  selectedArea: SelectedArea | null;
  onClose: () => void;
};

export default function IntelligencePanel({
  selectedArea,
  onClose
}: Props) {

  // =====================================================
  // STATE
  // =====================================================

  const [activeTab, setActiveTab] =
    useState("overview");

  const [chatInput, setChatInput] =
    useState("");

  const [chatMessages, setChatMessages] =
    useState<ChatMessage[]>([]);

  const [chatLoading, setChatLoading] =
    useState(false);

  const compactMode =
    activeTab !== "overview";

  // =====================================================
  // EARLY RETURN
  // =====================================================

  if (!selectedArea) {

    return null;
  }

  // =====================================================
  // DATA
  // =====================================================

  const metrics =
    selectedArea?.metrics || {};

  const context =
    selectedArea?.context || {};

  const location =
    context.location_name ||
    "Chicago Urban Zone";

  const overview =
    selectedArea.overview ||
    "Urban intelligence unavailable.";

  const safety =
    metrics?.safety_score || {};

  const theft =
    metrics?.theft_risk || {};

  const violence =
    metrics?.violence_risk || {};

  const nightlife =
    metrics?.nightlife_activity || {};

  const vitality =
    metrics?.urban_vitality || {};

  const workplace =
    metrics?.workplace_activity || {};

  const transit =
    metrics?.transit_access || {};

  const personality =
    metrics?.urban_personality?.value ||
    "Urban Zone";

  // =====================================================
  // AMBIENT COLOR SYSTEM
  // =====================================================

  const safetyValue =
    Number(safety.value || 0);

  const safetyLabel =
    safetyValue <= 40
      ? "Higher caution"
      : safetyValue <= 70
        ? "Moderate caution"
        : "Lower observed pressure";

  let ambientRGB = "";

  if (safetyValue <= 40) {

    ambientRGB =
      "248,113,113";

  } else if (safetyValue <= 70) {

    ambientRGB =
      "251,191,36";

  } else {

    ambientRGB =
      "52,211,153";
  }

  const ambientColor =
    `rgba(${ambientRGB},0.16)`;

  const ambientBorder =
    `rgba(${ambientRGB},0.22)`;

  const ambientText =
    `rgb(${ambientRGB})`;

  // =====================================================
  // CHAT
  // =====================================================

  async function sendChat() {

    if (
      !chatInput.trim() ||
      !selectedArea.context_id
    ) {
      return;
    }

    const question = chatInput;

    setChatMessages(prev => [

      ...prev,

      {
        role: "user",
        content: question
      }
    ]);

    setChatInput("");

    setChatLoading(true);

    try {

      const response = await fetch(

        apiUrl("/area-chat"),

        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json"
          },

          body: JSON.stringify({

            context_id:
              selectedArea.context_id,

            question
          })
        }
      );

      const data =
        await response.json();

      setChatMessages(prev => [

        ...prev,

        {
          role: "assistant",
          content:
            data.response ||
            "Urban intelligence unavailable."
        }
      ]);

    } catch {

      setChatMessages(prev => [

        ...prev,

        {
          role: "assistant",
          content:
            "Urban intelligence unavailable."
        }
      ]);

    } finally {

      setChatLoading(false);
    }
  }

  // =====================================================
  // UI
  // =====================================================

  return (

    <div className="
      fixed
      inset-x-2
      bottom-2
      z-[3000]
      top-auto

      sm:absolute
      sm:inset-x-auto
      sm:top-4
      sm:bottom-4
      sm:right-[72px]
    ">

      <motion.div

        initial={{
          opacity: 0,
          x: 20,
          scale: 0.985
        }}

        animate={{
          opacity: 1,
          x: 0,
          scale: 1
        }}

        transition={{
          duration: 0.24
        }}

        style={{

          boxShadow: `
            0 0 140px ${ambientColor}
          `,

          borderColor:
            ambientBorder
        }}

        className="
          relative

          w-[calc(100vw-16px)]
          max-w-none

          h-[min(70dvh,720px)]
          max-h-[70dvh]

          sm:w-[360px]
          md:w-[390px]
          lg:w-[420px]
          sm:max-w-[92vw]
          sm:h-full
          sm:max-h-[calc(100dvh-32px)]

          rounded-[30px]
          sm:rounded-[34px]

          border

          bg-[#0B131B]/88

          backdrop-blur-[30px]

          text-white

          flex
          flex-col

          overflow-hidden
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

        {/* AMBIENT OVERLAY */}

        <div

          style={{
            background: `
              radial-gradient(
                circle at top,
                ${ambientColor},
                transparent 70%
              )
            `
          }}

          className="
            absolute
            inset-0

            pointer-events-none

            opacity-80
          "
        />

        {/* HEADER */}

        <motion.div

          animate={{

            paddingTop:
              compactMode ? 10 : 20,

            paddingBottom:
              compactMode ? 8 : 0
          }}

          transition={{
            duration: 0.24
          }}

          className="
            relative

            px-4
            sm:px-5
            pt-7
            sm:pt-0
            shrink-0
          "
        >

          <div className="
            flex
            items-start
            justify-between
          ">

            <div className="flex-1">

              {/* OVERVIEW HEADER */}

              {!compactMode && (

                <>

                  <p className="
                    text-[10px]
                    uppercase
                    tracking-[0.35em]
                    text-cyan-200/45
                  ">

                    Urban Intelligence

                  </p>

                  <motion.h1

                    animate={{
                      opacity: 1,
                      y: 0
                    }}

                    className="
                      mt-3

                      text-[21px]
                      sm:text-[24px]

                      leading-tight
                      font-semibold

                      max-w-[220px]
                      sm:max-w-[260px]
                    "
                  >

                    {location}

                  </motion.h1>

                  <motion.p

                    initial={{
                      opacity: 0
                    }}

                    animate={{
                      opacity: 1
                    }}

                    className="
                      mt-2
                      text-[13px]
                      sm:text-[14px]
                      text-cyan-300
                    "
                  >

                    {personality}

                  </motion.p>

                </>

              )}

              {/* COMPACT AMBIENT STRIP */}

              {compactMode && (

                <motion.div

                  initial={{
                    opacity: 0,
                    scaleX: 0.7
                  }}

                  animate={{
                    opacity: 1,
                    scaleX: 1
                  }}

                  transition={{
                    duration: 0.25
                  }}

                  style={{
                    background: `
                      linear-gradient(
                        90deg,
                        transparent,
                        ${ambientColor},
                        transparent
                      )
                    `
                  }}

                  className="
                    mt-3

                    h-[7px]
                    w-[160px]
                    sm:w-[220px]

                    rounded-full

                    opacity-90

                    shadow-[0_0_25px_rgba(255,255,255,0.08)]
                  "
                />

              )}

            </div>

            {/* CLOSE BUTTON */}

            <button

              onClick={onClose}

              className="
                relative

                w-10
                h-10

                rounded-full

                bg-white/[0.05]

                border
                border-white/10

                flex
                items-center
                justify-center

                transition-all

                hover:bg-white/[0.08]
              "
            >

              <X size={18} />

            </button>

          </div>

          {/* HERO */}

          {!compactMode && (

            <motion.div

              initial={{
                opacity: 0,
                y: 10
              }}

              animate={{
                opacity: 1,
                y: 0
              }}

              transition={{
                duration: 0.22
              }}

              className="
                mt-4
                sm:mt-5
              "
            >

              <div

                style={{
                  borderColor:
                    ambientBorder
                }}

                className="
                  rounded-[28px]

                  p-4
                  sm:p-5

                  border

                  bg-gradient-to-br
                  from-white/[0.03]
                  to-transparent
                "
              >

                <div className="
                  flex
                  items-start
                  gap-4
                ">

                  <div

                    style={{
                      background:
                        ambientColor
                    }}

                    className="
                      w-12
                      h-12
                      sm:w-14
                      sm:h-14

                      rounded-[20px]

                      flex
                      items-center
                      justify-center
                    "
                  >

                    <Shield
                      size={28}
                      className="text-white"
                    />

                  </div>

                  <div className="min-w-0">

                    <p className="
                      text-[12px]
                      text-white/50
                    ">

                      Safety Score

                    </p>

                    <div className="
                      flex
                      items-end
                      gap-2
                    ">

                      <h2

                        style={{
                          color: ambientText
                        }}

                        className="
                          text-[44px]
                          sm:text-[54px]
                          leading-none
                          font-bold
                        "
                      >

                        {safety.value || 0}

                      </h2>

                      <span className="
                        pb-1.5
                        text-[14px]
                        sm:pb-2
                        sm:text-[16px]
                        text-white/55
                      ">

                        /100

                      </span>

                    </div>

                    <div className="
                      mt-3
                      inline-flex
                      items-center
                      gap-2
                      rounded-full
                      border
                      border-white/10
                      bg-white/[0.04]
                      px-3
                      py-1.5
                      text-[11px]
                      text-white/62
                    ">
                      <span
                        style={{
                          background: ambientText
                        }}
                        className="
                          h-2
                          w-2
                          rounded-full
                          shadow-[0_0_12px_rgba(255,255,255,0.22)]
                        "
                      />
                      {safetyLabel}
                    </div>

                    <p className="
                      mt-3
                      max-w-[240px]
                      text-[12px]
                      leading-relaxed
                      text-white/45
                    ">
                      {location} shows a {personality.toLowerCase()} profile with this score acting as the main safety anchor.
                    </p>
                  </div>

                </div>

              </div>

            </motion.div>

          )}

          {/* TABS */}

          <div className={`
            relative

            flex

            rounded-[18px]

            bg-white/[0.04]

            p-1

            ${
              compactMode

              ? "mt-4"

              : "mt-5"
            }
          `}>

            {[
              { key: "overview", label: "summary" },
              { key: "signals", label: "signals" },
              { key: "timeline", label: "timeline" },
              { key: "ai", label: "ask" }
            ].map((tab) => (

              <button
                key={tab.key}

                onClick={() =>
                  setActiveTab(tab.key)
                }

                className={`
                  relative

                  flex-1

                  py-3
                  sm:py-2.5

                  rounded-[14px]

                  text-[12px]
                  sm:text-[13px]

                  capitalize

                  transition-all
                  duration-200

                  ${
                    activeTab === tab.key

                    ? `
                      bg-white

                      text-black

                      shadow-[0_6px_18px_rgba(255,255,255,0.08)]
                    `

                    : `
                      text-white/42

                      hover:text-white/65
                    `
                  }
                `}
              >

                {tab.label}

              </button>
            ))}

          </div>

        </motion.div>

        {/* CONTENT */}

        <div className="
          relative

          flex-1
          min-h-0

          overflow-y-auto no-scrollbar

          px-4
          pb-[calc(env(safe-area-inset-bottom)+14px)]
          mt-4
          sm:px-5
          sm:pb-5
          sm:mt-5
        ">

          {selectedArea.loading && (

            <div className="
              rounded-[28px]

              border
              border-white/10

              bg-white/[0.03]

              p-5

              animate-pulse
            ">

              <div className="
                h-5
                w-44

                rounded-full

                bg-white/10
              " />

              <div className="
                mt-6
                space-y-3
              ">

                <div className="
                  h-4
                  rounded-full
                  bg-white/10
                " />

                <div className="
                  h-4
                  rounded-full
                  bg-white/10
                " />

                <div className="
                  h-4
                  w-[70%]

                  rounded-full
                  bg-white/10
                " />

              </div>

              <div className="
                mt-8

                grid
                grid-cols-2
                gap-4
              ">

                {[1,2,3,4].map((x) => (

                  <div
                    key={x}

                    className="
                      h-[110px]

                      rounded-[22px]

                      bg-white/[0.04]

                      border
                      border-white/10
                    "
                  />

                ))}

              </div>

            </div>
          )}

          {!selectedArea.loading && (

            <>

              {activeTab === "overview" && (

                <OverviewTab
                  overview={overview}
                />
              )}

              {activeTab === "signals" && (

                <SignalsTab

                  theft={theft}

                  violence={violence}

                  nightlife={nightlife}

                  transit={transit}

                  vitality={vitality}

                  workplace={workplace}
                />
              )}

              {activeTab === "timeline" && (

                <TimelineTab
                  metrics={metrics}
                />
              )}

              {activeTab === "ai" && (

                <AITab

                  chatMessages={chatMessages}

                  chatInput={chatInput}

                  setChatInput={setChatInput}

                  sendChat={sendChat}

                  chatLoading={chatLoading}
                />
              )}

            </>

          )}

        </div>

      </motion.div>

    </div>
  );
}
