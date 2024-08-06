import { useSignal } from "@preact/signals";
import { useEffect, useRef } from "preact/hooks";

const BASE_URL = import.meta.env.BASE_URL;

export default function IntroScene({ onEnded }) {
  const videoRef = useRef<HTMLVideoElement>();
  const isMuted = useSignal(true);
  const isLoaded = useSignal(false);

  useEffect(() => {
    const videoElement = videoRef.current;
    if (videoElement) {
      videoElement.play();
      videoElement.addEventListener("loadeddata", () => (isLoaded.value = true));
    }
  }, []);

  useEffect(() => {
    const onKeyPress = (ev: KeyboardEvent) => {
      if (ev.code === "Escape" || ev.code === "Enter") onEnded();
    };
    window.addEventListener("keydown", onKeyPress);
    return () => window.removeEventListener("keydown", onKeyPress);
  }, []);

  return (
    <div class="relative w-full h-full">
      <video ref={videoRef} class="bg-black h-full w-full" muted={isMuted.value} onEnded={onEnded}>
        <source src={`${BASE_URL}/intro.mp4`} type="video/mp4" />
      </video>
      {isLoaded.value && (
        <>
          <button
            class="absolute left-1 bottom-0 p-2 text-white opacity-75 hover:opacity-100 text-lg"
            onClick={() => (isMuted.value = !isMuted.value)}
          >
            {isMuted.value ? <i class="fa-solid fa-volume-xmark" /> : <i class="fa-solid fa-volume-high"></i>}
          </button>
          <button
            class="absolute right-1 bottom-0 p-2 text-white opacity-75 hover:opacity-100 text-lg"
            onClick={onEnded}
          >
            SKIP <i class="fa-solid fa-forward"></i>
          </button>
        </>
      )}
    </div>
  );
}
