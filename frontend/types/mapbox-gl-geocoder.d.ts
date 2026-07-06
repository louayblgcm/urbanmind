declare module "@mapbox/mapbox-gl-geocoder" {
  export default class MapboxGeocoder {
    constructor(options: Record<string, unknown>);
    onAdd(map: unknown): HTMLElement;
    onRemove(): void;
  }
}
