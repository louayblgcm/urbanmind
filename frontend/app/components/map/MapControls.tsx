"use client";

import { useMap } from "react-leaflet";

export default function MapControls() {

  const map = useMap();

  return (

    <div className="absolute bottom-6 right-6 z-[3000] flex flex-col gap-3">

      {/* ZOOM IN */}

      <button

        onClick={() =>
          map.zoomIn()
        }

        className="w-[52px] h-[52px] rounded-2xl backdrop-blur-3xl bg-[#07111f]/55 border border-white/10 text-white text-2xl shadow-2xl"
      >

        +

      </button>

      {/* ZOOM OUT */}

      <button

        onClick={() =>
          map.zoomOut()
        }

        className="w-[52px] h-[52px] rounded-2xl backdrop-blur-3xl bg-[#07111f]/55 border border-white/10 text-white text-2xl shadow-2xl"
      >

        −

      </button>

    </div>
  );
}