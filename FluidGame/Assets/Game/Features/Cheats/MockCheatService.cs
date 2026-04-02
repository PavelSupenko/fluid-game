using System;
using System.Collections.Generic;
using MeltIt.Services.Cheats;

namespace MeltIt.Features.Cheats
{
    public class MockCheatService : ICheatService
    {
        public event Action<CheatActionInfo>? ActionRegistered;
        public event Action<CheatActionInfo>? ActionUnregistered;
        public event Action<CheatPropertyInfo>? PropertyRegistered;
        public event Action<CheatPropertyInfo>? PropertyUnregistered;

        public List<CheatActionInfo> Actions { get; } = new();
        public List<CheatPropertyInfo> Properties { get; } = new();

        public BoolCheatProperty TestDevice { get; } = new(false);

        public void RegisterAction(string commandName, Action action, string? category = null) { }
        public void AttachAction(string commandName, Action action) { }
        public void DetachAction(string commandName, Action action) { }
        public void UnregisterAction(string commandName) { }
        public void UnregisterAction(Action action) { }
        public void CreateBindProperty(object binder, string currentLevel, Func<int> getter, Action<int> setter, 
            string? category = null, int? min = null, int? max = null) { }
        public void CreateBindProperty(object binder, string name, Func<float> getter, Action<float> setter,
            string? category = null, float? min = null, float? max = null) { }
        public void CreateBindProperty(object binder, string name, Func<string> getter, Action<string> setter,
            string? category = null) { }
        public void CreateBindProperty(object binder, string name, Func<bool> getter, Action<bool> setter,
            string? category = null) { }
        public void CreateBindProperty<TEnum>(object binder, string name, Func<TEnum> getter, Action<TEnum> setter,
            string? category = null) 
            where TEnum : struct, Enum { }
        public void CreateBindProperty<TArray>(object binder, string name, TArray[] array, 
            Func<TArray> getter, Action<TArray> setter, string? category = null) { }

        public void CreateBindProperty(object binder, string name, Func<TimeSpan> getter, Action<TimeSpan> setter, 
            string category = null) { }

        public void UnregisterAllProperties(object binder) { }
        public void UnregisterProperty(object binder, string name) { }
        public void AttachPropertyCallback(string name, Action action) { }
        public void DetachPropertyCallback(string name, Action action) { }
    }
}