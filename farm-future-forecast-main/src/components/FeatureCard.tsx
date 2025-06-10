
import React from 'react';
import { LucideIcon } from 'lucide-react';

interface FeatureCardProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  delay?: number;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ 
  title, 
  description, 
  icon,
  delay = 0
}) => {
  return (
    <div 
      className="feature-card bg-white rounded-xl p-6 shadow-md hover:shadow-xl border border-gray-100 transition-all duration-300 hover:scale-105"
      style={{ 
        animationDelay: `${delay}ms`,
        opacity: 0,
        animation: `fadeIn 0.5s ease-out ${delay}ms forwards`
      }}
    >
      <div className="flex items-center justify-center w-14 h-14 mb-4 bg-gradient-to-br from-ag-green-light/20 to-ag-green/20 rounded-full">
        {icon}
      </div>
      
      <h3 className="text-xl font-semibold text-ag-earth mb-2">{title}</h3>
      <p className="text-gray-600">{description}</p>
    </div>
  );
};

export default FeatureCard;
