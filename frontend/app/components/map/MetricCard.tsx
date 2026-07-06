"use client";

import { motion } from "framer-motion";

type Props = {
  title: string;
  value: string;
  icon: React.ReactNode;
  color: keyof typeof styles;
};

const styles = {
  red: {
    text: "text-red-300",
    bg: "bg-red-400/10",
    border: "border-red-400/20",
    glow: "shadow-[0_0_22px_rgba(248,113,113,0.06)]",
  },
  orange: {
    text: "text-orange-300",
    bg: "bg-orange-400/10",
    border: "border-orange-400/20",
    glow: "shadow-[0_0_22px_rgba(251,146,60,0.06)]",
  },
  emerald: {
    text: "text-emerald-300",
    bg: "bg-emerald-400/10",
    border: "border-emerald-400/20",
    glow: "shadow-[0_0_22px_rgba(52,211,153,0.06)]",
  },
  purple: {
    text: "text-purple-300",
    bg: "bg-purple-400/10",
    border: "border-purple-400/20",
    glow: "shadow-[0_0_22px_rgba(192,132,252,0.06)]",
  },
  cyan: {
    text: "text-cyan-300",
    bg: "bg-cyan-400/10",
    border: "border-cyan-400/20",
    glow: "shadow-[0_0_22px_rgba(34,211,238,0.06)]",
  },
} as const;

export default function MetricCard({
  title,
  value,
  icon,
  color
}: Props) {

  const s = styles[color];

  return (

    <motion.div

      initial={{
        opacity: 0,
        y: 14,
        scale: 0.985
      }}

      animate={{
        opacity: 1,
        y: 0,
        scale: 1
      }}

      transition={{
        duration: 0.28,
        ease: "easeOut"
      }}

      whileHover={{
        y: -3,
        scale: 1.01
      }}

      className={`
        relative

        overflow-hidden

        rounded-[24px]

        border
        ${s.border}

        bg-white/[0.028]

        px-4
        py-4
        sm:px-4
        sm:py-3.5

        transition-all
        duration-300

        ${s.glow}

        hover:bg-white/[0.04]
      `}
    >

      {/* SOFT BACKGROUND GLOW */}

      <div className={`
        absolute
        inset-0

        opacity-[0.025]

        ${s.bg}
      `} />

      {/* ICON */}

      <motion.div

        whileHover={{
          rotate: -3,
          scale: 1.06
        }}

        transition={{
          duration: 0.2
        }}

        className={`
          relative

        w-10
        h-10
        sm:w-9
        sm:h-9

          rounded-[14px]

          ${s.bg}

          flex
          items-center
          justify-center

          ${s.text}
        `}
      >

        {icon}

      </motion.div>

      {/* TITLE */}

      <p className="
        relative

        mt-3.5

        text-[10px]
        sm:text-[11px]

        tracking-[0.08em]

        text-white/35

        font-medium
      ">

        {title}

      </p>

      {/* VALUE */}

      <motion.h3

        initial={{
          opacity: 0,
          y: 6
        }}

        animate={{
          opacity: 1,
          y: 0
        }}

        transition={{
          duration: 0.35,
          delay: 0.03
        }}

        className={`
          relative

          mt-2

          text-[28px]
          sm:text-[32px]

          leading-none

          font-semibold

          tracking-tight

          ${s.text}
        `}
      >

        {value}

      </motion.h3>

      {/* SUBTLE BOTTOM SHINE */}

      <div className="
        absolute

        bottom-0
        left-0
        right-0

        h-[1px]

        bg-gradient-to-r
        from-transparent
        via-white/10
        to-transparent
      " />

    </motion.div>
  );
}
