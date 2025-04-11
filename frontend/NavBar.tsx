

const NavBar = () => {
  const handlePredictNow = () => {
    window.location.href = 'http://localhost:8503';
  };

  return (
    <nav className="w-full py-4">
      <div className="container mx-auto flex justify-between items-center px-4 md:px-6">
        <div className="flex items-center space-x-2">
          <span className="text-ag-green font-bold text-2xl">🌿BazaarBataye</span>
          <div className="hidden md:flex h-6 w-[1px] bg-ag-green/30 mx-2"></div>
          <span className="hidden md:inline-block text-sm text-ag-earth/80">Agriculture Market Price Predictor</span>
        </div>
        
        <button 
          onClick={handlePredictNow}
          className="cta-button bg-gradient-to-r from-ag-green to-ag-green-dark text-white px-4 py-2 rounded-full font-medium 
                     flex items-center gap-2 hover:shadow-lg hover:shadow-ag-green/20 transition-all duration-300"
        >
          <span>🤖AgriBot</span>
        </button>
      </div>
    </nav>
  );
};

export default NavBar;
