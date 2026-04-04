using MeltIt.Services.Cheats;
using MeltIt.Features.Inject;
using VContainer.Unity;
using VContainer;
using UnityEngine.UI;
using UnityEngine;

#if DebugLog
using MobileConsole;
#endif

namespace MeltIt.Features.Cheats
{
    public class SettingsConsoleViewOpener : MonoBehaviour
    {
        [SerializeField] private Button _button = null!;
#if DebugLog
        private ICheatService _cheatService;
        private CheatView _cheatView = null!;

        private void OnEnable()=> 
            _button.onClick.AddListener(OpenSettingsView);

        private void OnDisable() => 
            _button.onClick.RemoveListener(OpenSettingsView);

        private void OnDestroy()
        {
            if (_cheatService != null)
                _cheatView.Dispose();
        }

        private void OpenSettingsView()
        {
            _cheatService ??= LifetimeScope.Find<GameLifetimeScope>().Container.Resolve<ICheatService>();
            _cheatView ??= new CheatView(_cheatService);

            LogConsole.PushSubView(_cheatView);
        }
#endif
    }
}