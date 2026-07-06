"use client";

import { useControl } from "react-map-gl/mapbox";

import MapboxGeocoder from "@mapbox/mapbox-gl-geocoder";

import "@mapbox/mapbox-gl-geocoder/dist/mapbox-gl-geocoder.css";

import "./searchbar.css";

type Props = {
  mapboxAccessToken?: string;
  // Retained for the unused legacy Leaflet map so it remains type-checkable.
  query?: string;
  setQuery?: (value: string) => void;
  currentCenter?: [number, number];
  onLocationSelect?: (lat: number, lon: number) => void;
  onClear?: () => void;
};

export default function SearchBar({

  mapboxAccessToken

}: Props) {

  useControl(() => {

    return new MapboxGeocoder({

      accessToken:
        mapboxAccessToken || process.env.NEXT_PUBLIC_MAPBOX_TOKEN || "",

      marker: false,

      collapsed: false,

      placeholder:
        "Search places, streets, areas...",

      types:
        "place,address,poi",

      language:
        "en",

      limit: 6,

      flyTo: {

        speed: 1.2,

        curve: 1.4,

        essential: true
      }
    });

  },

  {
    position: "top-left"
  });

  return null;
}
