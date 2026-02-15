import { motion } from 'framer-motion';

const EmotionCard = ({ emotion, icon: Icon, onClick, isSelected }) => {
    return (
        <motion.button
            whileHover={{ scale: 1.05, y: -5 }}
            whileTap={{ scale: 0.95 }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            onClick={() => onClick(emotion.name)}
            className={`relative p-6 rounded-2xl flex flex-col items-center justify-center gap-4 transition-all duration-300 w-full aspect-square group
        ${isSelected
                    ? 'glass-card bg-white/20 border-white/40 shadow-[0_0_20px_rgba(255,255,255,0.2)]'
                    : 'glass-card hover:bg-white/10'
                }
      `}
        >
            <div className={`p-4 rounded-full bg-gradient-to-br ${emotion.color} text-white shadow-lg group-hover:shadow-xl transition-all`}>
                <Icon size={32} />
            </div>
            <span className="font-semibold text-lg tracking-wide uppercase text-white/90 group-hover:text-white">
                {emotion.name}
            </span>

            {isSelected && (
                <motion.div
                    layoutId="outline"
                    className="absolute inset-0 rounded-2xl border-2 border-white/50"
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                />
            )}
        </motion.button>
    );
};

export default EmotionCard;
