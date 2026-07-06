"use client";
import IntelligencePanel from "./IntelligencePanel";
import MapControls from "./MapControls";
import {
  MapContainer,
  TileLayer,
  Marker,
  Rectangle,
  useMap,useMapEvents
} from "react-leaflet";
import LayerToggle from "./LayerToggle";
import L from "leaflet";
import {
  useState,
  useEffect,
} from "react";

import SearchBar from "./SearchBar";
import { apiUrl } from "@/app/lib/api";
const customIcon = new L.Icon({
  iconUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",

  shadowUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",

  iconSize: [25, 41],

  iconAnchor: [12, 41],
});
type FlyToProps = {
  position: [number, number];
};

function FlyToLocation({
  position,
}: FlyToProps) {

  const map = useMap();

  useEffect(() => {

    map.flyTo(position, 15);

  }, [position, map]);

  return null;
}
function TrackMap({
  onMove,
}: {
  onMove: (
    center: [number, number]
  ) => void;
}) {

  const map = useMap();

  map.on("moveend", () => {

    const center = map.getCenter();

    onMove([
      center.lat,
      center.lng,
    ]);
  });

  return null;
}
function getLayerColor(
  cell: any,
  activeLayer: string
) {

  let value = 0;

  if (activeLayer === "safety") {

    value =
      100 - cell.safety_score;

  }

  else if (
    activeLayer ===
    "vehicle_theft"
  ) {

    value =
      cell.vehicle_theft_risk * 100;
  }

  else if (
    activeLayer ===
    "night"
  ) {

    value =
      cell.night_risk_ratio * 100;
  }

  else if (
    activeLayer ===
    "density"
  ) {

    value =
      cell.relative_density;
  }

  if (value < 20) {
    return "#22c55e";
  }

  if (value < 40) {
    return "#84cc16";
  }

  if (value < 60) {
    return "#facc15";
  }

  if (value < 80) {
    return "#f97316";
  }

  return "#ef4444";
}
function findNearestCell(

  lat: number,

  lon: number,

  cells: any[]

) {

  let nearest = null;

  let minDistance = Infinity;

  for (const cell of cells) {

    const dLat =
      lat - cell.lat_cell;

    const dLon =
      lon - cell.lon_cell;

    const distance =
      Math.sqrt(
        dLat * dLat
        +
        dLon * dLon
      );

    if (distance < minDistance) {

      minDistance = distance;

      nearest = cell;
    }
  }

  return nearest;
}
function MapClickHandler({

  onMapClick

}: {

  onMapClick: (
    lat: number,
    lon: number
  ) => void;
}) {

  useMapEvents({

    click(e) {

      onMapClick(
        e.latlng.lat,
        e.latlng.lng
      );
    }
  });

  return null;
}
async function reverseGeocode(

  lat: number,

  lon: number

) {

  try {

    const response =
      await fetch(

`https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lon}`,

{
  headers: {
    "Accept-Language":
      "en"
  }
}
);

    if (!response.ok) {

      return "";
    }

    const data =
      await response.json();

    return (
      data.display_name
      ||
      ""
    );

  } catch {

    return "";
  }
}
export default function Map() {

  const [position, setPosition] =
    useState<[number, number] | null>(null);
  const [gridCells, setGridCells] =
  useState<any[]>([]);
  const [activeLayer, setActiveLayer] =
  useState("none");
  const [selectedCell, setSelectedCell] =
  useState<any>(null);
  const [searchQuery, setSearchQuery] =
  useState("");
  const [currentCenter, setCurrentCenter] =
  useState<[number, number]>([
    41.8781,
    -87.6298,
  ]);
  useEffect(() => {

  async function fetchGrid() {

    const response = await fetch(
      apiUrl("/grid-cells")
    );

    const data = await response.json();

    setGridCells(data);
  }

  fetchGrid();

}, []);


  return (
    <div className="relative w-full h-screen">

      <SearchBar
          query={searchQuery}

          setQuery={setSearchQuery}
          currentCenter={currentCenter}

          onLocationSelect={(lat, lon) => {

            setPosition([lat, lon]);

            const nearestCell = findNearestCell(
                lat,
                lon,
                gridCells
            );

            setSelectedCell(
                nearestCell
            );

          }}   onClear={() => {

  setPosition(null);

  setSelectedCell(null);
}}
      />

      <MapContainer
        center={[41.8781, -87.6298]}
        zoom={11}
        className="w-full h-full"
        zoomControl={false}
      ><TrackMap
  onMove={(center) => {
    setCurrentCenter(center);
  }}
/>
        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />


{gridCells.map((cell, index) => {

  const CELL_SIZE = 0.002;

  return (

    <Rectangle
      key={index}

      interactive={true}

      eventHandlers={{

        click: () => {

          setSelectedCell(cell);
        }
      }}

      bounds={[
        [
          cell.lat_cell,
          cell.lon_cell
        ],
        [
          cell.lat_cell + CELL_SIZE,
          cell.lon_cell + CELL_SIZE
        ]
      ]}

      pathOptions={{

        fillColor: getLayerColor(
          cell,
          activeLayer
        ),

        fillOpacity:

          activeLayer === "none"

          ? 0

          : 0.22,

        color:

          activeLayer === "none"

          ? "transparent"

          : "#111111",

        weight: 0.3,
      }}
    />
  );
})}
              {position && (
          <>
            <Marker
              position={position}
              icon={customIcon}
            />

            <FlyToLocation
              position={position}
            />
          </>
        )}
      <MapClickHandler

  onMapClick={async (

    lat,

    lon

  ) => {

    setPosition([
      lat,
      lon
    ]);

    const nearestCell =
      findNearestCell(
        lat,
        lon,
        gridCells
      );

    setSelectedCell(
      nearestCell
    );

    const address =
      await reverseGeocode(
        lat,
        lon
      );

    setSearchQuery(
      address
    );
  }}
/><MapControls />
      </MapContainer>
      <LayerToggle



  activeLayer={activeLayer}

  setActiveLayer={setActiveLayer}

/>
        <IntelligencePanel
  selectedCell={selectedCell}
/>


    </div>
  );
}
