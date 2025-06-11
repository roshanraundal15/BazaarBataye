const Footer = () => {
  return (
    <footer className="bg-ag-earth py-12 text-white">
      <div className="container mx-auto px-4 md:px-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Left Column: GitHub Links */}
          <div>
            <h4 className="text-lg font-semibold mb-4 text-ag-gold-light">GitHub</h4>
            <ul className="space-y-2">
              <li><a href="https://github.com/roshanraundal15" className="text-white/80 hover:text-white transition-colors">Roshan Raundal</a></li>
              <li><a href="https://github.com/karanpanchallll" className="text-white/80 hover:text-white transition-colors">Karan Panchal</a></li>
              <li><a href="https://github.com/mayursapkal41" className="text-white/80 hover:text-white transition-colors">Mayur Sapkal</a></li>
              
            </ul>
          </div>

          {/* Middle Column: Contact Info */}
          <div>
            <h4 className="text-lg font-semibold mb-4 text-ag-gold-light">Contact</h4>
            <address className="not-italic space-y-2">
              <p className="text-white/80">raundal2001@gmail.com</p>
              <p className="text-white/80">panchalkaran@gmail.com</p>
              <p className="text-white/80">mayursapkal41@gmail.com</p>
              
            </address>
          </div>

          {/* Right Column: YouTube Video */}
          <div>
            
            <div className="aspect-w-16 aspect-h-9">
              <iframe
                className="rounded-xl w-full h-48 md:h-60"
                src="https://youtu.be/yYJNuWuqjG8"
                title="Bazaar Bataye Demo"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              ></iframe>
            </div>
          </div>
        </div>

        {/* Description below GitHub + Contact */}
        <div className="mt-12 text-center">
          <h3 className="text-xl font-bold mb-2 text-ag-gold-light">BazaarBataye</h3>
          <p className="text-white/80">
            Empowering farmers with AI-powered crop price predictions and market insights.
          </p>
        </div>

        {/* Bottom Footer */}
        <div className="mt-8 pt-8 border-t border-white/10 text-center text-white/60">
          <p>&copy; {new Date().getFullYear()} BazaarBataye: Agriculture Market Price Predictor. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
