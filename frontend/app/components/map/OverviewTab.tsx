import { Sparkles } from "lucide-react";

type OverviewTabProps = {
  overview: string;
};

export default function OverviewTab({
  overview
}: OverviewTabProps) {

  return (

    <div className="
      rounded-[26px]

      border
      border-white/10

      bg-white/[0.03]

      p-5
    ">

      <div className="
        flex
        items-center
        justify-between
        gap-2
      ">
        <div className="
          flex
          items-center
          gap-2
        ">
          <Sparkles
            size={18}
            className="text-cyan-300"
          />

          <h2 className="
            text-[15px]
            font-medium
          ">
            Area Summary
          </h2>
        </div>

        <span className="
          rounded-full
          border
          border-cyan-300/15
          bg-cyan-400/8
          px-2.5
          py-1
          text-[10px]
          uppercase
          tracking-[0.18em]
          text-cyan-200/60
        ">
          Human-readable
        </span>

      </div>

      <p className="
        mt-4

        text-[14px]
        sm:text-[14px]

        leading-[1.85]

        text-white/82
        sm:max-w-[96%]
      ">

        {overview}

      </p>

    </div>
  );
}
