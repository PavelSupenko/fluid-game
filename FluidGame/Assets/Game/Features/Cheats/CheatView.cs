#if DebugLog
using MeltIt.Services.Cheats;
using System.Collections.Generic;
using MobileConsole.UI;
using UnityEngine;
using System;

namespace MeltIt.Features.Cheats
{
    public class CheatView : ViewBuilder, IDisposable
    {
        private readonly Color[] _categoryColors =
        {
            new (0.6f, 0.8f, 0.6f), // Light Sage Green
            new (0.7f, 0.7f, 0.9f), // Soft Lavender
            new (0.8f, 0.6f, 0.7f), // Muted Rose
            new (0.6f, 0.8f, 0.8f), // Pale Turquoise
            new (0.9f, 0.75f, 0.6f), // Warm Peach
            new (0.7f, 0.6f, 0.8f), // Gentle Violet
            new (0.8f, 0.8f, 0.6f), // Soft Gold
            new (0.6f, 0.7f, 0.9f), // Sky Blue
            new (0.9f, 0.6f, 0.6f), // Light Coral
            new (0.7f, 0.9f, 0.7f), // Mint Green
            new (0.85f, 0.7f, 0.85f), // Dusty Lilac
            new (0.75f, 0.85f, 0.9f), // Powder Blue
            new (0.9f, 0.8f, 0.7f), // Creamy Orange
            new (0.65f, 0.85f, 0.65f), // Avocado Green
            new (0.8f, 0.75f, 0.65f)  // Sandy Beige
        };
        
        private readonly ICheatService _cheatService;
        private readonly Dictionary<CheatActionInfo, NodeView> _actionNodes = new();
        private readonly Dictionary<string, CheatPropertyView> _propertyViews = new();
        private readonly Dictionary<string, CategoryNodeView> _customSections = new();
        private CategoryNodeView _commonCustomSection;

        public CheatView(ICheatService cheatService)
        {
            _cheatService = cheatService;
            closeAllSubViewOnAction = false;
            title = "WingPlay Settings";
            
            CreateMainControls();
            _cheatService.ActionRegistered += OnActionRegistered;
            _cheatService.ActionUnregistered += OnActionUnregistered;
            _cheatService.PropertyRegistered += OnPropertyRegistered;
            _cheatService.PropertyUnregistered += OnPropertyUnregistered;
        }

        public void Dispose()
        {
            _cheatService.ActionRegistered -= OnActionRegistered;
            _cheatService.ActionUnregistered -= OnActionUnregistered;
            _cheatService.PropertyRegistered -= OnPropertyRegistered;
            _cheatService.PropertyUnregistered -= OnPropertyUnregistered;
        }

        public override void OnPrepareToShow()
        {
            base.OnPrepareToShow();
            UpdateProperties();
        }

        private void CreateMainControls()
        {
            // TODO: temporary disabled. Set it only from backoffice
            // OnPropertyRegistered(new CheatPropertyInfo
            // {
            //     Name = Constants.Cheats.Properties.IsTestDevice, 
            //     Category = Constants.Cheats.Category.Player,
            //     Property = _cheatService.TestDevice
            // });

            string contextCheatsCategoryName = "Context Actions and Variables";
            _commonCustomSection = CreateCategory(contextCheatsCategoryName);
            _commonCustomSection.iconColor = GetRandomColor(contextCheatsCategoryName);
            foreach (var actionInfo in _cheatService.Actions) 
                OnActionRegistered(actionInfo);
            
            foreach (var propertyInfo in _cheatService.Properties)
                OnPropertyRegistered(propertyInfo);
        }

        private void OnActionRegistered(CheatActionInfo actionInfo)
        {
            CategoryNodeView parentCategory = GetParentCategory(actionInfo.Category, _commonCustomSection);
            NodeView node = AddButton(actionInfo.Name, 
                _ =>
                {
                    foreach (Action action in actionInfo.Actions) 
                        action();

                    UpdateProperties();
                }, 
                parentCategory);
            
            _actionNodes[actionInfo] = node;
            
            parentCategory.ToggleExpand();
            parentCategory.ToggleExpand();
            Rebuild();
        }

        private void OnActionUnregistered(CheatActionInfo actionInfo)
        {
            if (_actionNodes.TryGetValue(actionInfo, out NodeView node))
            {
                node.RemoveFromParent();
                RemoveEmptyCategories();
                Rebuild();
            }
        }

        private void OnPropertyRegistered(CheatPropertyInfo propertyInfo)
        {
            CategoryNodeView parentCategory = GetParentCategory(propertyInfo.Category, _commonCustomSection);
            CheatPropertyView view = new CheatPropertyView(propertyInfo.Name, propertyInfo.Property, 
                this, parentCategory);
            
            _propertyViews[propertyInfo.Name] = view;
            propertyInfo.Property.ValueChanged += UpdatePropertyViews;
            
            parentCategory.ToggleExpand();
            parentCategory.ToggleExpand();
            Rebuild();
        }

        private void OnPropertyUnregistered(CheatPropertyInfo propertyInfo)
        {
            if (_propertyViews.TryGetValue(propertyInfo.Name, out CheatPropertyView view))
            {
                if (view.MainNode != null)
                {
                    view.Property.ValueChanged -= UpdatePropertyViews;
                    view.MainNode.RemoveFromParent();
                    RemoveEmptyCategories();
                    Rebuild();
                }
            }
        }

        private CategoryNodeView GetParentCategory(string category, CategoryNodeView parentCategory)
        {
            if (category != null)
            {
                if (_customSections.TryGetValue(category, out CategoryNodeView customSection) == false)
                {
                    customSection = CreateCategory(category);
                    customSection.iconColor = GetRandomColor(category);
                    customSection.ToggleExpand();
                    _customSections[category] = customSection;
                }

                parentCategory = customSection;
            }

            return parentCategory;
        }

        private void RemoveEmptyCategories()
        {
            List<KeyValuePair<string, CategoryNodeView>> emptyCategories = new();
            foreach (var category in _customSections)
            {
                if (category.Value.children.Count == 0)
                    emptyCategories.Add(category);
            }

            foreach (var category in emptyCategories)
            {
                category.Value.RemoveFromParent();
                _customSections.Remove(category.Key);
            }
        }

        private void UpdateProperties()
        {
            foreach (CheatPropertyView propertyView in _propertyViews.Values)
            {
                propertyView.UpdateProperty();
                propertyView.UpdateView();
            }
            
            Rebuild();
        }
        
        private void UpdatePropertyViews()
        {
            foreach (CheatPropertyView propertyView in _propertyViews.Values) 
                propertyView.UpdateView();
            
            Rebuild();
        }
        
        private Color GetRandomColor(string name)
        {
            int index = Mathf.Abs(name.GetHashCode()) % _categoryColors.Length;
            return _categoryColors[index];
        }
    }
}
#endif