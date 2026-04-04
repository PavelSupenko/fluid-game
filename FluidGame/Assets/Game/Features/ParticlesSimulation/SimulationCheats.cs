using System;
using ParticlesSimulation.Debug;
using MeltIt.Services.Cheats;
using UnityEngine;
using VContainer;

namespace DefaultNamespace
{
    public class SimulationCheats : MonoBehaviour
    {
        private DebugParticleMode? _debugParticleModeOverride;
        private ICheatService _cheatService;

        [SerializeField] private DebugParticleRenderController _renderController;

        [Inject]
        public void Inject(ICheatService cheatService)
        {
            _cheatService = cheatService;
        }

        private void Start()
        {
            if (_cheatService == null)
                return;
            
            _cheatService.CreateBindProperty(this, "Render Mode", 
                () => _debugParticleModeOverride ?? DebugParticleMode.Normal, 
                mode => _debugParticleModeOverride = mode);
        }

        private void Update()
        {
            if (_debugParticleModeOverride.HasValue)
                _renderController.Mode = _debugParticleModeOverride.Value;
        }
    }
}