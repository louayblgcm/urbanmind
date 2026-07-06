import {
  Car,
  Moon,
  Train,
  AlertTriangle,
  Sparkles,
  BriefcaseBusiness
} from "lucide-react";

import MetricCard from "./MetricCard";

type SignalMetric = {
  percentage?: number | string;
};

type SignalsTabProps = {
  theft?: SignalMetric;
  violence?: SignalMetric;
  nightlife?: SignalMetric;
  transit?: SignalMetric;
  vitality?: SignalMetric;
  workplace?: SignalMetric;
};

export default function SignalsTab({
  theft,
  violence,
  nightlife,
  transit,
  vitality,
  workplace
}: SignalsTabProps) {

  return (

    <div className="
      grid
      grid-cols-2
      gap-3
      sm:gap-3.5
    ">

      <MetricCard
        title="Theft Risk"
        value={`${Number(
          theft?.percentage || 0
        ).toFixed(1)}%`}
        icon={<Car size={18} />}
        color="red"
      />

      <MetricCard
        title="Violent Crime"
        value={`${Number(
          violence?.percentage || 0
        ).toFixed(1)}%`}
        icon={<AlertTriangle size={18} />}
        color="orange"
      />

      <MetricCard
        title="Evening Activity"
        value={`${Number(
          nightlife?.percentage || 0
        ).toFixed(1)}%`}
        icon={<Moon size={18} />}
        color="purple"
      />

      <MetricCard
        title="Transit Access"
        value={`${Number(
          transit?.percentage || 0
        ).toFixed(1)}%`}
        icon={<Train size={18} />}
        color="cyan"
      />

      <MetricCard
        title="Street Activity"
        value={`${Number(
          vitality?.percentage || 0
        ).toFixed(1)}%`}
        icon={<Sparkles size={18} />}
        color="emerald"
      />

      <MetricCard
        title="Workday Activity"
        value={`${Number(
          workplace?.percentage || 0
        ).toFixed(1)}%`}
        icon={<BriefcaseBusiness size={18} />}
        color="cyan"
      />
    </div>
  );
}
