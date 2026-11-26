import { useCallback, useEffect, useState } from "react";

export interface Async<T> { data: T | null; error: unknown; loading: boolean; reload: () => void }

/** Run an async loader when `deps` change; keeps the last good data while reloading. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): Async<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let live = true;
    setLoading(true);
    fn().then((d) => { if (live) { setData(d); setError(null); } })
      .catch((e) => { if (live) setError(e); })
      .finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload };
}

/** Current hash route, e.g. "#/alert/lab-ae-3" -> ["alert", "lab-ae-3"]. */
export function useRoute(): string[] {
  const read = () => window.location.hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent).filter(Boolean);
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export function go(...parts: string[]) {
  window.location.hash = "#/" + parts.map(encodeURIComponent).join("/");
}
