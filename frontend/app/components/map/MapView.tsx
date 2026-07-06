"use client";
import ActivityFeedPanel from "./ActivityFeedPanel";

import IntelligencePanel from "./IntelligencePanel";
import "./global.css";
import SearchBar from "./SearchBar";
import { apiUrl } from "@/app/lib/api";

import Map, {

  Marker,

  NavigationControl,

  GeolocateControl,

  FullscreenControl,

  ScaleControl,

  Source,

  Layer

} from "react-map-gl/mapbox";

import "mapbox-gl/dist/mapbox-gl.css";

import {

  useState,

  useRef

} from "react";

type SelectedArea = {
  loading?: boolean;
  location?: {
    lat: number;
    lon: number;
  };
  [key: string]: unknown;
} | null;

export default function MapView() {

  const [selectedArea, setSelectedArea] =
    useState<SelectedArea>(null);

  const [mobilePanel, setMobilePanel] =
    useState<"insights" | "feed">(
      "insights"
    );

  // =====================================================
  // ABORT CONTROLLER
  // =====================================================

  const currentRequest =
    useRef<AbortController | null>(null);

  // =====================================================
  // REQUEST ID
  // =====================================================

  const latestRequestId =
    useRef(0);

  return (

    <div
      data-area-selected={
        selectedArea
          ? "true"
          : "false"
      }
      className="map-shell relative w-full h-screen overflow-visible"
    >

      <Map

        mapboxAccessToken={
          process.env.NEXT_PUBLIC_MAPBOX_TOKEN
        }

        initialViewState={{

          longitude: -87.6298,

          latitude: 41.8781,

          zoom: 13.5,

          pitch: 60,

          bearing: -25
        }}

        style={{
          width: "100%",
          height: "100%"
        }}

        mapStyle="mapbox://styles/mapbox/standard"

        antialias={true}

        reuseMaps

        renderWorldCopies={false}

        dragRotate={true}

        touchZoomRotate={true}

        doubleClickZoom={true}

        maxPitch={85}

        onClick={async (event) => {

          const {
            lng,
            lat
          } = event.lngLat;

          try {

            // =====================================
            // CANCEL OLD REQUEST
            // =====================================

            if (currentRequest.current) {

              currentRequest.current.abort();
            }

            // =====================================
            // NEW CONTROLLER
            // =====================================

            const controller =
              new AbortController();

            currentRequest.current =
              controller;

            // =====================================
            // REQUEST ID
            // =====================================

            const requestId =
              ++latestRequestId.current;

            // =====================================
            // IMMEDIATE UI FEEDBACK
            // =====================================

            setSelectedArea({

              loading: true,

              location: {

                lat,

                lon: lng
              }
            });

            // =====================================
            // FETCH OVERVIEW
            // =====================================

            const response =
              await fetch(

                apiUrl("/area-overview"),

                {

                  method: "POST",

                  headers: {
                    "Content-Type":
                      "application/json"
                  },

                  signal:
                    controller.signal,

                  body: JSON.stringify({

                    lat,

                    lon: lng
                  })
                }
              );

            const data =
              await response.json();

            // =====================================
            // IGNORE STALE REQUESTS
            // =====================================

            if (
              requestId !==
              latestRequestId.current
            ) {
              return;
            }

            // =====================================
            // UPDATE
            // =====================================

            setSelectedArea({

              ...data,

              loading: false,

              location: {

                lat,

                lon: lng
              }
            });

            setMobilePanel(
              "insights"
            );

          } catch (error: unknown) {

            // =====================================
            // IGNORE ABORT ERRORS
            // =====================================

            if (
              !(
                error instanceof DOMException &&
                error.name === "AbortError"
              )
            ) {

              console.error(
                "FETCH ERROR:",
                error
              );
            }
          }
        }}
      >

        {/* SEARCH */}

        {/* SEARCH */}

{!selectedArea && (

  <SearchBar

    mapboxAccessToken={
      process.env.NEXT_PUBLIC_MAPBOX_TOKEN!
    }
  />

)}

        {/* CONTROLS */}

        <NavigationControl
          position="bottom-right"
        />

        <GeolocateControl
          position="bottom-right"
          trackUserLocation={true}
        />

        <FullscreenControl
          position="bottom-right"
        />

        <ScaleControl />

        {/* ===================================== */}
        {/* SELECTED BLOCK */}
        {/* ===================================== */}

        {selectedArea &&
          selectedArea.location && (

          <Source

            id="selected-cell"

            type="geojson"

            data={{

              type: "Feature",

              geometry: {

                type: "Polygon",

                coordinates: [[

                  [

                    selectedArea.location.lon - 0.00075,

                    selectedArea.location.lat - 0.00075
                  ],

                  [

                    selectedArea.location.lon + 0.00075,

                    selectedArea.location.lat - 0.00075
                  ],

                  [

                    selectedArea.location.lon + 0.00075,

                    selectedArea.location.lat + 0.00075
                  ],

                  [

                    selectedArea.location.lon - 0.00075,

                    selectedArea.location.lat + 0.00075
                  ],

                  [

                    selectedArea.location.lon - 0.00075,

                    selectedArea.location.lat - 0.00075
                  ]
                ]]
              },

              properties: {}
            }}
          >

            {/* GLOW */}

            <Layer

              id="selected-cell-fill"

              type="fill"

              paint={{

                "fill-color":
                  "#3B82F6",

                "fill-opacity": 0.12
              }}
            />

            {/* BORDER */}

            <Layer

              id="selected-cell-border"

              type="line"

              paint={{

                "line-color":
                  "#9CCBFF",

                "line-width": 3,

                "line-opacity": 0.95
              }}
            />

          </Source>
        )}

        {/* ===================================== */}
        {/* MARKER */}
        {/* ===================================== */}

        {selectedArea &&
          selectedArea.location && (

          <Marker

            longitude={
              selectedArea.location.lon
            }

            latitude={
              selectedArea.location.lat
            }

            anchor="bottom"
          >

            <div className="relative">

              {/* GLOW */}

              <div className="absolute left-1/2 top-1/2 w-14 h-14 -translate-x-1/2 -translate-y-1/2 rounded-full bg-blue-400/30 blur-2xl" />

              {/* PIN */}

              <div className="w-8 h-8 bg-blue-500 rounded-full border-[4px] border-white shadow-[0_0_25px_rgba(59,130,246,0.85)]" />

            </div>

          </Marker>
        )}

      </Map>

      {selectedArea && (

        <div className="
          fixed
          left-1/2
          top-[calc(env(safe-area-inset-top)+12px)]
          z-[3200]
          -translate-x-1/2
          rounded-full
          border
          border-white/10
          bg-[#081017]/88
          p-1
          backdrop-blur-xl
          sm:hidden
        ">

          {[
            {
              key: "insights",
              label: "Insights"
            },
            {
              key: "feed",
              label: "Live Feed"
            }
          ].map((item) => (

            <button
              key={item.key}
              onClick={() =>
                setMobilePanel(
                  item.key as
                    | "insights"
                    | "feed"
                )
              }
              className={`
                rounded-full
                px-4
                py-2
                text-[12px]
                font-medium
                transition-all
                ${
                  mobilePanel === item.key
                    ? `
                      bg-white
                      text-black
                    `
                    : `
                      text-white/58
                    `
                }
              `}
            >

              {item.label}

            </button>
          ))}

        </div>
      )}

      {selectedArea && (

        <div className="hidden sm:block">
          <ActivityFeedPanel
            selectedArea={selectedArea}
            onClose={() =>
              setSelectedArea(null)
            }
          />
        </div>
      )}

      {selectedArea && (

        <div className="hidden sm:block">
          <IntelligencePanel
            selectedArea={selectedArea}
            onClose={() =>
              setSelectedArea(null)
            }
          />
        </div>
      )}

      {selectedArea &&
        mobilePanel === "feed" && (
          <div className="sm:hidden">
            <ActivityFeedPanel
              selectedArea={selectedArea}
              onClose={() =>
                setSelectedArea(null)
              }
            />
          </div>
        )}

      {selectedArea &&
        mobilePanel ===
          "insights" && (
          <div className="sm:hidden">
            <IntelligencePanel
              selectedArea={selectedArea}
              onClose={() =>
                setSelectedArea(null)
              }
            />
          </div>
        )}

    </div>
  );
}
