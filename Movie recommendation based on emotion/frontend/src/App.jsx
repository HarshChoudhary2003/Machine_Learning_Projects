import { useState } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Smile, Frown, Zap, Ghost, Trash2, Heart, Search, Monitor, Cloud
} from 'lucide-react';
import EmotionCard from './components/EmotionCard';
import MovieCard from './components/MovieCard';
import clsx from 'clsx';

const emotions = [
  { name: 'Happy', icon: Smile, color: 'from-orange-400 to-yellow-500' },
  { name: 'Sad', icon: Frown, color: 'from-blue-500 to-indigo-700' },
  { name: 'Excited', icon: Zap, color: 'from-yellow-400 to-red-500' },
  { name: 'Scared', icon: Ghost, color: 'from-purple-800 to-black' },
  { name: 'Angry', icon: Trash2, color: 'from-red-600 to-orange-700' },
  { name: 'Romantic', icon: Heart, color: 'from-pink-400 to-rose-600' },
  { name: 'Bored', icon: Search, color: 'from-gray-400 to-slate-600' },
  { name: 'Curious', icon: Cloud, color: 'from-sky-300 to-cyan-500' }
];

function App() {
  const [selectedEmotion, setSelectedEmotion] = useState(null);
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleEmotionSelect = async (emotion) => {
    setSelectedEmotion(emotion);
    setLoading(true);
    setMovies([]);
    setError(null);

    try {
      const response = await axios.get(`http://localhost:8000/recommend?emotion=${emotion}`);
      setMovies(response.data);
    } catch (err) {
      setError('Failed to fetch recommendations. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white font-sans overflow-hidden relative">
      <div className="absolute inset-0 z-0">
        <div className="absolute top-[-50%] left-[-50%] w-[200%] h-[200%] bg-gradient-radial from-blue-900/20 via-slate-900/50 to-slate-900 animate-pulse-slow pointer-events-none" />
        <div className="absolute bottom-[-20%] right-[-20%] w-[100%] h-[100%] bg-gradient-radial from-purple-900/20 via-slate-900/50 to-slate-900 animate-pulse-slow pointer-events-none delay-1000" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6 py-12 md:py-20 flex flex-col gap-12">
        <header className="text-center space-y-4">
          <motion.h1
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-5xl md:text-7xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 drop-shadow-sm"
          >
            MoodFlix
          </motion.h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-xl text-slate-400 max-w-2xl mx-auto"
          >
            Discover the perfect movie for your current mood. Select how you're feeling and let AI do the rest.
          </motion.p>
        </header>

        <section className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-6 max-w-4xl mx-auto w-full">
          {emotions.map((emotion, index) => (
            <motion.div
              key={emotion.name}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.05 }}
            >
              <EmotionCard
                emotion={emotion}
                icon={emotion.icon}
                onClick={() => handleEmotionSelect(emotion.name)}
                isSelected={selectedEmotion === emotion.name}
              />
            </motion.div>
          ))}
        </section>

        <AnimatePresence mode="wait">
          {loading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center py-20"
            >
              <div className="w-16 h-16 border-4 border-t-purple-500 border-white/20 rounded-full animate-spin mb-4" />
              <p className="text-lg text-slate-300 animate-pulse">Consulting the cinema gods...</p>
            </motion.div>
          )}

          {error && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="text-center py-10 text-red-400 bg-red-900/20 rounded-xl border border-red-500/20 max-w-lg mx-auto"
            >
              <p>{error}</p>
            </motion.div>
          )}

          {!loading && movies.length > 0 && (
            <motion.div
              layout
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 50 }}
              className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8 w-full"
            >
              {movies.map((movie, index) => (
                <MovieCard key={index} movie={movie} index={index} />
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

export default App;
