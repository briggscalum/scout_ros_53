// Generated by gencpp from file scout_msgs/ScoutLightState.msg
// DO NOT EDIT!


#ifndef SCOUT_MSGS_MESSAGE_SCOUTLIGHTSTATE_H
#define SCOUT_MSGS_MESSAGE_SCOUTLIGHTSTATE_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace scout_msgs
{
template <class ContainerAllocator>
struct ScoutLightState_
{
  typedef ScoutLightState_<ContainerAllocator> Type;

  ScoutLightState_()
    : mode(0)
    , custom_value(0)  {
    }
  ScoutLightState_(const ContainerAllocator& _alloc)
    : mode(0)
    , custom_value(0)  {
  (void)_alloc;
    }



   typedef uint8_t _mode_type;
  _mode_type mode;

   typedef uint8_t _custom_value_type;
  _custom_value_type custom_value;





  typedef boost::shared_ptr< ::scout_msgs::ScoutLightState_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::scout_msgs::ScoutLightState_<ContainerAllocator> const> ConstPtr;

}; // struct ScoutLightState_

typedef ::scout_msgs::ScoutLightState_<std::allocator<void> > ScoutLightState;

typedef boost::shared_ptr< ::scout_msgs::ScoutLightState > ScoutLightStatePtr;
typedef boost::shared_ptr< ::scout_msgs::ScoutLightState const> ScoutLightStateConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::scout_msgs::ScoutLightState_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::scout_msgs::ScoutLightState_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::scout_msgs::ScoutLightState_<ContainerAllocator1> & lhs, const ::scout_msgs::ScoutLightState_<ContainerAllocator2> & rhs)
{
  return lhs.mode == rhs.mode &&
    lhs.custom_value == rhs.custom_value;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::scout_msgs::ScoutLightState_<ContainerAllocator1> & lhs, const ::scout_msgs::ScoutLightState_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace scout_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::scout_msgs::ScoutLightState_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::scout_msgs::ScoutLightState_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::scout_msgs::ScoutLightState_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::scout_msgs::ScoutLightState_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::scout_msgs::ScoutLightState_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::scout_msgs::ScoutLightState_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::scout_msgs::ScoutLightState_<ContainerAllocator> >
{
  static const char* value()
  {
    return "51866248399dda20e62f6b250914288e";
  }

  static const char* value(const ::scout_msgs::ScoutLightState_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x51866248399dda20ULL;
  static const uint64_t static_value2 = 0xe62f6b250914288eULL;
};

template<class ContainerAllocator>
struct DataType< ::scout_msgs::ScoutLightState_<ContainerAllocator> >
{
  static const char* value()
  {
    return "scout_msgs/ScoutLightState";
  }

  static const char* value(const ::scout_msgs::ScoutLightState_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::scout_msgs::ScoutLightState_<ContainerAllocator> >
{
  static const char* value()
  {
    return "uint8 mode\n"
"uint8 custom_value\n"
;
  }

  static const char* value(const ::scout_msgs::ScoutLightState_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::scout_msgs::ScoutLightState_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.mode);
      stream.next(m.custom_value);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ScoutLightState_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::scout_msgs::ScoutLightState_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::scout_msgs::ScoutLightState_<ContainerAllocator>& v)
  {
    s << indent << "mode: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.mode);
    s << indent << "custom_value: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.custom_value);
  }
};

} // namespace message_operations
} // namespace ros

#endif // SCOUT_MSGS_MESSAGE_SCOUTLIGHTSTATE_H
