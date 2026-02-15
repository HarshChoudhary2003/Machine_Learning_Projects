import { motion } from 'framer-motion';
import { Star, Calendar } from 'lucide-react';

const MovieCard = ({ movie, index }) => {
    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
            whileHover={{ y: -10, scale: 1.02 }}
            className="relative group overflow-hidden rounded-xl shadow-2xl glass-card transition-all w-full max-w-sm aspect-[2/3] mx-auto"
        >
            <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent z-10 opacity-60 group-hover:opacity-80 transition-opacity" />

            <img
                src={movie.image ? movie.image : "https://via.placeholder.com/300x450?text=No+Image"}
                alt={movie.title}
                loading="lazy"
                className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
            />

            <div className="absolute bottom-0 left-0 w-full p-6 z-20 translate-y-4 group-hover:translate-y-0 transition-transform duration-300">
                <h3 className="text-2xl font-bold text-white mb-2 line-clamp-2 drop-shadow-md leading-tight">
                    {movie.title}
                </h3>

                <div className="flex items-center gap-4 text-sm text-gray-300 font-medium">
                    {movie.rating && (
                        <div className="flex items-center gap-1 bg-yellow-500/20 px-2 py-1 rounded-md backdrop-blur-sm border border-yellow-500/30">
                            <Star className="text-yellow-400 fill-yellow-400" size={14} />
                            <span>{movie.rating}</span>
                        </div>
                    )}

                    {movie.year && (
                        <div className="flex items-center gap-1 bg-blue-500/20 px-2 py-1 rounded-md backdrop-blur-sm border border-blue-500/30">
                            <Calendar className="text-blue-400" size={14} />
                            <span>{movie.year}</span>
                        </div>
                    )}
                </div>

                <motion.a
                    href={movie.url || `https://www.imdb.com/find?q=${encodeURIComponent(movie.title)}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="mt-4 block w-full py-2 bg-white/20 hover:bg-white/30 text-center rounded-lg backdrop-blur-md border border-white/20 font-semibold transition-colors opacity-0 group-hover:opacity-100 translate-y-4 group-hover:translate-y-0 duration-300 delay-100"
                >
                    View on IMDb
                </motion.a>
            </div>
        </motion.div>
    );
};

export default MovieCard;
