using System.Collections.Generic;
using System;

namespace MeltIt.Services.Cheats
{
    public class CheatActionInfo
    {
        public string Name { get; set; }
        public List<Action> Actions { get; set; }
        public string? Category { get; set; }
    }
    
    public class CheatPropertyInfo
    {
        public string Name { get; set; }
        public ICheatProperty Property { get; set; }
        public string? Category { get; set; }
    }
    
    public interface ICheatService
    {
        event Action<CheatActionInfo> ActionRegistered;
        event Action<CheatActionInfo> ActionUnregistered;

        event Action<CheatPropertyInfo> PropertyRegistered;
        event Action<CheatPropertyInfo> PropertyUnregistered;

        BoolCheatProperty TestDevice { get; }

        List<CheatActionInfo> Actions { get; }
        List<CheatPropertyInfo> Properties { get; }
        
        void RegisterAction(string commandName, Action action, string? category = null);
        void UnregisterAction(Action action);
        void UnregisterAction(string commandName);
        
        /// <summary> Attach the new callback to the already existed action by command name </summary>
        void AttachAction(string commandName, Action action);
        void DetachAction(string commandName, Action action);

        void CreateBindProperty(object binder, string name, Func<int> getter, Action<int> setter, 
            string? category = null, int? min = null, int? max = null);
        void CreateBindProperty(object binder, string name, Func<float> getter, Action<float> setter, 
            string? category = null, float? min = null, float? max = null);
        void CreateBindProperty(object binder, string name, Func<string> getter, Action<string> setter,
            string? category = null);
        void CreateBindProperty(object binder, string name, Func<bool> getter, Action<bool> setter,
            string? category = null);
        void CreateBindProperty<TEnum>(object binder, string name, Func<TEnum> getter, Action<TEnum> setter,
            string? category = null) where TEnum : struct, Enum;
        void CreateBindProperty<TArray>(object binder, string name, TArray[] array, 
            Func<TArray> getter, Action<TArray> setter, string? category = null);
        void CreateBindProperty(object binder, string name, Func<TimeSpan> getter, Action<TimeSpan> setter, 
            string? category = null);

        void UnregisterAllProperties(object binder);
        void UnregisterProperty(object binder, string name);
        void AttachPropertyCallback(string name, Action action);
        void DetachPropertyCallback(string name, Action action);
    }
}