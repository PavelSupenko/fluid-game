#if DebugLog && FALSE
using MeltIt.Features.Gameplay;
using VContainer.Unity;
using MobileConsole;
using System;

namespace MeltIt.Features.Cheats
{
    public class LogConsoleOpenListener : IInitializable, IDisposable
    {
        private readonly IShapeMovingService _shapeMovingService;

        public LogConsoleOpenListener(IShapeMovingService shapeMovingService)
        {
            _shapeMovingService = shapeMovingService;
        }
        
        public void Initialize()
        {
            LogConsole.OnVisibilityChanged += OnLogConsoleVisibilityChanged;
        }

        public void Dispose()
        {
            LogConsole.OnVisibilityChanged -= OnLogConsoleVisibilityChanged;
        }
        
        private void OnLogConsoleVisibilityChanged(bool isConsoleActive)
        {
            if (isConsoleActive)
                _shapeMovingService.Disable();
            else
                _shapeMovingService.Enable();
        }
    }
}
#endif