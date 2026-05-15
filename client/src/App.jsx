import { useCallback, useEffect, useRef, useState } from "react";

const API = "/api";

function comfortClass(comfort) {
    return comfort?.toLowerCase().replace(/\s+/g, "-") ?? "";
}

async function api(path, method = "GET") {
    const res = await fetch(`${API}${path}`, { method });
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    return res.json();
}

export default function App() {
    const [status, setStatus] = useState(null);
    const [error, setError] = useState(null);
    const [autoRun, setAutoRun] = useState(false);
    const intervalRef = useRef(null);

    const loadStatus = useCallback(async () => {
        try {
            const data = await api("/status");
            setStatus(data);
            setError(null);
        } catch {
            setError(
                "Cannot reach server. Start it from server/: uvicorn app.main:app --reload",
            );
        }
    }, []);

    const reset = async () => {
        try {
            const data = await api("/reset", "POST");
            setStatus(data);
            setError(null);
            setAutoRun(true);
        } catch {
            setError("Failed to reset simulation.");
        }
    };

    const step = useCallback(async () => {
        try {
            const data = await api("/step", "POST");
            setStatus(data);
            setError(null);
            if (!data.running) setAutoRun(false);
        } catch {
            setError("Failed to run simulation step.");
        }
    }, []);

    useEffect(() => {
        loadStatus();
    }, [loadStatus]);

    useEffect(() => {
        if (!autoRun) {
            clearInterval(intervalRef.current);
            return;
        }
        intervalRef.current = setInterval(step, 2000);
        return () => clearInterval(intervalRef.current);
    }, [autoRun, step]);

    return (
        <div>
            <h1>Smart Classroom</h1>
            <p className="subtitle">RL-powered energy control simulation</p>

            <div className="controls">
                <button onClick={reset}>Start / Reset</button>
                <button
                    className="secondary"
                    onClick={step}
                    disabled={!status?.running}
                >
                    Step
                </button>
                <button
                    className="secondary"
                    onClick={() => setAutoRun((v) => !v)}
                    disabled={!status?.running}
                >
                    {autoRun ? "Pause" : "Auto Run"}
                </button>
            </div>

            {error && <div className="banner error">{error}</div>}
            {status && !status.policy_loaded && (
                <div className="banner">
                    No trained policy found. Run <code>python train.py</code> in
                    server/ai for full RL decisions.
                </div>
            )}
            {status && !status.running && status.step > 0 && (
                <div className="banner done">
                    Simulation complete — {status.step} steps, total reward{" "}
                    {status.total_reward}
                </div>
            )}

            {status && (
                <div className="grid">
                    <div className="card">
                        <h2>Classroom</h2>
                        <div className="metric">
                            {status.students}
                            <small>students present</small>
                        </div>
                        <div className="metric" style={{ marginTop: "1rem" }}>
                            {status.temperature}°C
                            <small>
                                <span
                                    className={`comfort ${comfortClass(status.comfort)}`}
                                >
                                    {status.comfort}
                                </span>
                            </small>
                        </div>
                    </div>

                    <div className="card">
                        <h2>Devices</h2>
                        <div className="device">
                            <span>Lights</span>
                            <span className={status.lights_on ? "on" : "off"}>
                                {status.lights_on ? "ON" : "OFF"}
                            </span>
                        </div>
                        <div className="device">
                            <span>AC</span>
                            <span className={status.ac_on ? "on" : "off"}>
                                {status.ac_on ? "ON" : "OFF"}
                            </span>
                        </div>
                        <div className="device">
                            <span>Fan speed</span>
                            <span>{status.fan_speed} / 3</span>
                        </div>
                    </div>

                    <div className="card">
                        <h2>RL decision</h2>
                        <div className="metric">
                            {status.action_name}
                            <small>step {status.step} / 50</small>
                        </div>
                    </div>

                    <div className="card">
                        <h2>Performance</h2>
                        <div className="device">
                            <span>Reward</span>
                            <span>{status.reward}</span>
                        </div>
                        <div className="device">
                            <span>Total reward</span>
                            <span>{status.total_reward}</span>
                        </div>
                        <div className="device">
                            <span>Energy (avg)</span>
                            <span>{status.average_energy}</span>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
