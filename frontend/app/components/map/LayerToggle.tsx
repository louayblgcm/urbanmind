"use client";

import { useState } from "react";

type Props = {

  activeLayer: string;

  setActiveLayer: (
    layer: string
  ) => void;
};

export default function LayerToggle({

  activeLayer,

  setActiveLayer

}: Props) {

  const [open, setOpen] =
    useState(false);

  const layers = [

    {
      id: "none",
      label: "Default View"
    },

    {
      id: "safety",
      label: "Safety Heatmap"
    },

    {
      id: "vehicle_theft",
      label: "Vehicle Theft Risk"
    },

    {
      id: "night",
      label: "Nighttime Risk"
    },

    {
      id: "density",
      label: "Crime Density"
    }
  ];

  const activeLabel =
    layers.find(
      (l) =>
        l.id === activeLayer
    )?.label;

  return (

    <div className="absolute top-4 right-4 z-[1000]">

      {/* BUTTON */}

      <button

        onClick={() =>
          setOpen(!open)
        }

          className="bg-white text-black shadow-2xl rounded-2xl px-5 py-3 text-sm font-medium flex items-center gap-3"      >

        {activeLabel}

        <span className="text-black">          ▼
        </span>

      </button>

      {/* MENU */}

      {open && (

<div className="absolute right-0 mt-2 bg-white rounded-2xl shadow-2xl overflow-hidden w-[240px]">
          {layers.map((layer) => (

            <button

              key={layer.id}

              onClick={() => {

                setActiveLayer(
                  layer.id
                );

                setOpen(false);
              }}

              className={`

                          w-full
                          text-left
                          px-5
                          py-3
                          text-sm
                          font-medium
                          transition-all
                          
                          ${
                            activeLayer === layer.id
                          
                            ? "bg-black text-white"
                          
                            : "bg-white text-black hover:bg-gray-100"
                          }
                          `}
            >

              {layer.label}

            </button>
          ))}

        </div>
      )}

    </div>
  );
}