import React, { useState } from "react";

const API_BASE = "http://localhost:3001/api";

export default function BookCover({ title, author, summary }) {
    const [loading, setLoading] = useState(false);
    const [coverImageUrl, setCoverImageUrl] = useState(null); // Va stoca URL-ul unei singure imagini
    const [error, setError] = useState(null);

    const generateCover = async () => {
        setLoading(true);
        setError(null);
        setCoverImageUrl(null); // Resetăm imaginea la fiecare generare

        try {
            const resp = await fetch(`${API_BASE}/cover`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ title, author, summary }),
            });
            const data = await resp.json();
            if (resp.ok && data.url) { // Presupunem că API-ul returnează direct un URL al imaginii
                setCoverImageUrl(data.url);
            } else {
                setError(data.error || "Eroare la generarea copertei.");
            }
        } catch (err) {
            setError("Nu am putut contacta serverul.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col items-center gap-4 p-4 border rounded-2xl shadow-lg bg-white flex-shrink-0">
            <h3 className="text-xl font-bold text-center">{title}</h3>
            <p className="text-gray-600">{author}</p>

            {/* Containerul pentru copertă - acum afișează o singură imagine */}
            <div className="relative w-full aspect-[2/3] bg-gray-100 rounded-lg overflow-hidden flex justify-center items-center">
                {coverImageUrl ? (
                    <img
                        src={coverImageUrl}
                        alt={`${title} cover`}
                        className="w-full h-full object-cover" // Asigură că imaginea umple spațiul
                    />
                ) : (
                    <button
                        onClick={generateCover}
                        disabled={loading}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
                    >
                        {loading ? "Generez..." : "Generează Copertă"}
                    </button>
                )}
            </div>

            {error && <p className="text-red-500 text-sm mt-2">{error}</p>}
        </div>
    );
}